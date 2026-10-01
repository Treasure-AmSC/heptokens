"""Datamodule for training classifiers directly on pre-computed VQ-VAE tokens.

The tokens are produced by ``scripts/export_tokens.py`` and stored in an ``.npz``
archive with the following arrays:

    indices     : [N, num_csts, num_quantizers]  int16  (-1 = masked constituent)
    labels      : [N]                            int8   (already mapped to 0..n_classes-1)
    codebooks   : [num_quantizers, codebook_size, codebook_dim]  float32
    eventNumber : [N]                            int64  (optional)

The continuous codebook vectors are reconstructed on the fly, per chunk, via

    z_q[i, j, :] = sum over q of codebooks[q, indices[i, j, q], :]

(masked positions are zeroed). This matches both ``scripts/sanity_check_tokens.py``
and the live VQ-VAE ``encode`` output (verified: ``layer.codebook == layer.embed.T``).

The resulting batches contain ``{"z_q", "mask", "labels"}`` and are consumed directly
by ``VectorClassifier`` / ``VectorEmbedder`` with ``model.tokenizer_ckpt=null`` (no
on-the-fly tokenization). Because batches carry no raw ``csts``/``jets``, this module
runs with ``transforms=None`` (``preprocess_batch`` is skipped entirely).

For large files (e.g. the 168M-jet ``large`` split, ~54 GB of int16 indices) the
indices/labels are read from memory-mapped uncompressed ``.npy`` sidecars when present,
so worker processes share the page cache instead of each copying the full array. Use
``extract_npy()`` once to create those sidecars next to the ``.npz``.
"""

import logging
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from heptokens.data.atlas_mappable import BaseMapModule
from heptokens.data.collation import collate_and_transform

log = logging.getLogger(__name__)


def _npy_sidecar_dir(npz_path: str, sidecar_root: str | None = None) -> Path:
    """Directory holding the uncompressed .npy sidecars for an .npz token file.

    By default the ``<stem>_npy`` directory sits next to the .npz. If ``sidecar_root`` is
    given, the same-named directory is looked up under that root instead. This lets the
    caller read sidecars that were staged to fast local storage (e.g. /lscratch NVMe)
    without changing the .npz path or any array content — purely an I/O-location change.
    """
    p = Path(npz_path)
    name = p.stem + "_npy"
    if sidecar_root is not None:
        return Path(sidecar_root) / name
    return p.with_name(name)


def _read_npy_header(fh):
    """Read an .npy stream header, returning (shape, fortran_order, dtype)."""
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fh)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(fh)
    return np.lib.format._read_array_header(fh, version)


def extract_npy(npz_path: str, overwrite: bool = False, row_chunk: int = 200_000) -> Path:
    """Extract indices/labels/codebooks from an .npz into uncompressed .npy sidecars.

    Run this ONCE per token file. Subsequent training reads the .npy files via mmap
    with bounded RAM.

    The extraction streams each array straight from the compressed zip member into a
    memory-mapped output array ``row_chunk`` rows at a time, so peak RAM stays bounded
    (a few hundred MB) even for the 54 GB ``large`` indices array — it never decompresses
    a full array into memory.

    Returns the sidecar directory.
    """
    out_dir = _npy_sidecar_dir(npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        members = {Path(name).stem: name for name in zf.namelist()}
        for key in ("indices", "labels", "codebooks"):
            dst = out_dir / f"{key}.npy"
            if dst.exists() and not overwrite:
                log.info(f"sidecar exists, skipping: {dst}")
                continue
            if key not in members:
                raise KeyError(f"{key} not found in {npz_path} (members: {list(members)})")
            log.info(f"streaming {dst} ...")
            with zf.open(members[key]) as src:
                shape, fortran_order, dtype = _read_npy_header(src)
                out = np.lib.format.open_memmap(
                    str(dst), mode="w+", dtype=dtype, shape=shape, fortran_order=fortran_order
                )
                n_rows = shape[0] if shape else 1
                row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
                row_bytes = row_elems * dtype.itemsize
                flat = out.reshape(n_rows, row_elems) if shape else out.reshape(1, 1)
                pos = 0
                while pos < n_rows:
                    c = min(row_chunk, n_rows - pos)
                    buf = src.read(c * row_bytes)
                    if len(buf) != c * row_bytes:
                        raise EOFError(
                            f"short read for {key}: got {len(buf)} expected {c * row_bytes}"
                        )
                    flat[pos : pos + c] = np.frombuffer(buf, dtype=dtype).reshape(c, row_elems)
                    pos += c
                out.flush()
                del out, flat
    log.info(f"extracted .npy sidecars to {out_dir}")
    return out_dir


class TokenNpzDataset(IterableDataset):
    """Streams pre-computed token indices, reconstructing z_q per chunk.

    Mirrors ``heptokens.data.atlas_iterable.IndexedIterMapDataset`` but reads from
    in-memory / memory-mapped numpy arrays of token indices rather than an HDF5 file.

    Supports one or more *segments* (e.g. the small/medium/large token files combined
    into a single ~200M-jet training set). Each segment carries its own
    ``indices``/``labels``/``codebooks`` arrays and the subset of row indices assigned to
    this split. Blocks from all segments are interleaved (and shuffled, for train) so
    batches mix the constituent files.
    """

    def __init__(
        self,
        segments: list[dict],
        num_csts: int | None = None,
        chunk_size: int = 1000,
        shuffle: bool = False,
        shuffle_buffer: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        # segments: list of {indices [N,num_csts,nq], labels [N], codebooks [nq,cb_size,cb_dim],
        #                    split_indices [M]} dicts (arrays possibly mmap'd).
        if not segments:
            raise ValueError("TokenNpzDataset requires at least one segment")
        self.segments = segments
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0

        first_cb = segments[0]["codebooks"]
        self.num_quantizers = first_cb.shape[0]
        self.cb_size = first_cb.shape[1]
        self.cb_dim = first_cb.shape[2]
        total_csts = segments[0]["indices"].shape[1]
        self.num_csts = total_csts if num_csts is None else min(total_csts, num_csts)
        self._length = int(sum(len(s["split_indices"]) for s in segments))

    def _reconstruct_zq(
        self, idx_block: np.ndarray, codebooks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """idx_block: [c, num_csts, nq] int -> (z_q [c, num_csts, cb_dim], mask [c, num_csts])."""
        c = idx_block.shape[0]
        z_q = np.zeros((c, self.num_csts, self.cb_dim), dtype=np.float32)
        for q in range(self.num_quantizers):
            q_idx = idx_block[:, :, q]  # [c, num_csts]
            valid = q_idx >= 0
            flat_idx = np.clip(q_idx, 0, self.cb_size - 1)
            gathered = codebooks[q][flat_idx]  # [c, num_csts, cb_dim]
            gathered[~valid] = 0.0
            z_q += gathered
        mask = idx_block[:, :, 0] >= 0  # [c, num_csts]  (True = real constituent)
        return z_q, mask

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Per-epoch RNG: seeded from (seed, worker_id, epoch) so the split stays
        # reproducible while the shuffle order varies across epochs.
        self._epoch += 1
        rng = None
        if self.shuffle:
            base = 0 if self.seed is None else self.seed
            rng = np.random.default_rng((base, worker_id, self._epoch))

        # Sort each segment's split for efficient (slice-able) contiguous reads, then
        # build a flat list of (segment_id, chunk_start) blocks across all segments.
        seg_sorted = [np.sort(s["split_indices"]) for s in self.segments]
        blocks = [
            (sid, pos)
            for sid, si in enumerate(seg_sorted)
            for pos in range(0, len(si), self.chunk_size)
        ]
        # Shard whole blocks across workers (disjoint, deterministic).
        blocks = blocks[worker_id::num_workers]
        if self.shuffle:
            rng.shuffle(blocks)

        buffer: list[dict] = []
        for sid, pos in blocks:
            si = seg_sorted[sid]
            seg = self.segments[sid]
            chunk_indices = si[pos : pos + self.chunk_size]
            idx_start, idx_end = int(chunk_indices[0]), int(chunk_indices[-1]) + 1
            if idx_end - idx_start == len(chunk_indices):
                sel = slice(idx_start, idx_end)
            else:
                sel = chunk_indices

            idx_block = np.asarray(seg["indices"][sel])[:, : self.num_csts, :]
            labels_block = np.asarray(seg["labels"][sel]).astype(np.int64)
            z_q, mask = self._reconstruct_zq(idx_block, seg["codebooks"])

            for i in range(idx_block.shape[0]):
                sample = {
                    "z_q": z_q[i],
                    "mask": mask[i],
                    "labels": labels_block[i],
                }
                if self.shuffle and self.shuffle_buffer > 0:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer:
                        j = int(rng.integers(len(buffer)))
                        buffer[j], buffer[-1] = buffer[-1], buffer[j]
                        yield buffer.pop()
                else:
                    yield sample

        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def __len__(self) -> int:
        return self._length


class TokenNpzModule(BaseMapModule):
    """DataModule that splits one or more token .npz files into train/val/test.

    Mirrors ``heptokens.data.atlas_iterable.SingleFileIterModule``: splits are computed
    in ``__init__`` from a seeded permutation, and three ``TokenNpzDataset`` streams are
    built. ``transforms`` is forced to ``None`` (tokens are already the final input).

    Pass a single ``data_path`` for one file, or ``data_paths`` (a list) to train on the
    union of several token files (e.g. small+medium+large ≈ 200M jets). Each file is split
    independently with the same ``train_frac``/``val_frac``/``test_frac`` (so the combined
    ratios are preserved) and the files must share an identical codebook.
    """

    def __init__(
        self,
        *,
        data_path: str | None = None,
        data_paths: list[str] | None = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        chunk_size: int = 1000,
        num_csts: int | None = None,
        shuffle_buffer: int = 0,
        # Optional local dir holding staged <stem>_npy sidecars (e.g. /lscratch). None =
        # read sidecars from next to the .npz (default; unchanged behaviour).
        sidecar_root: str | None = None,
        # codebook_dim is accepted for config symmetry with the model; not used here.
        codebook_dim: int | None = None,
        **kwargs,
    ) -> None:
        # Tokens carry no raw csts/jets, so preprocessing must be skipped.
        kwargs["transforms"] = None
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        paths = self._resolve_paths(data_path, data_paths)
        self.data_paths = paths
        self.sidecar_root = sidecar_root
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.chunk_size = chunk_size

        train_segs: list[dict] = []
        val_segs: list[dict] = []
        test_segs: list[dict] = []
        codebooks_ref: np.ndarray | None = None
        total_jets = 0

        for path in paths:
            indices, labels, codebooks = self._load_arrays(path, sidecar_root)
            if codebooks_ref is None:
                codebooks_ref = codebooks
            elif not np.array_equal(codebooks_ref, codebooks):
                raise ValueError(
                    f"Codebook mismatch: {path} has a different codebook than the first "
                    f"file. All token files combined into one run must share a codebook."
                )

            n = indices.shape[0]
            total_jets += n
            train_size = int(n * train_frac)
            val_size = int(n * val_frac)
            if seed is not None:
                perm = np.random.default_rng(seed).permutation(n)
            else:
                perm = np.arange(n)
            train_idx = perm[:train_size]
            val_idx = perm[train_size : train_size + val_size]
            test_idx = perm[train_size + val_size :]

            base = dict(indices=indices, labels=labels, codebooks=codebooks)
            train_segs.append({**base, "split_indices": train_idx})
            val_segs.append({**base, "split_indices": val_idx})
            test_segs.append({**base, "split_indices": test_idx})

            log.info(
                f"TokenNpzModule file {Path(path).name}: {n:,} jets, "
                f"indices {indices.shape} {indices.dtype} "
                f"(source: {'mmap .npy' if isinstance(indices, np.memmap) else '.npz in RAM'})"
            )

        self.codebooks = codebooks_ref
        log.info(
            f"TokenNpzModule: {len(paths)} file(s), {total_jets:,} jets total, "
            f"codebooks {codebooks_ref.shape}"
        )

        ds_kwargs = dict(num_csts=num_csts, chunk_size=chunk_size)
        self.train_set = TokenNpzDataset(
            train_segs, shuffle=True, shuffle_buffer=shuffle_buffer, seed=seed, **ds_kwargs
        )
        self.valid_set = TokenNpzDataset(val_segs, **ds_kwargs)
        self.test_set = TokenNpzDataset(test_segs, **ds_kwargs)

    @staticmethod
    def _resolve_paths(data_path: str | None, data_paths: list[str] | None) -> list[str]:
        """Return the list of token .npz paths, preferring ``data_paths`` when given."""
        if data_paths:
            return list(data_paths)
        if data_path:
            return [data_path]
        raise ValueError("TokenNpzModule requires either data_path or data_paths")

    @staticmethod
    def _load_arrays(data_path: str, sidecar_root: str | None = None):
        """Return (indices, labels, codebooks). Prefer mmap'd .npy sidecars if present."""
        sidecar = _npy_sidecar_dir(data_path, sidecar_root)
        if sidecar_root is not None and not (sidecar / "indices.npy").exists():
            raise FileNotFoundError(
                f"sidecar_root={sidecar_root} was given but staged sidecars are missing: "
                f"{sidecar}/indices.npy. Stage the <stem>_npy dir there first."
            )
        if (sidecar / "indices.npy").exists():
            indices = np.load(sidecar / "indices.npy", mmap_mode="r")
            labels = np.load(sidecar / "labels.npy", mmap_mode="r")
            codebooks = np.load(sidecar / "codebooks.npy")  # small, keep in RAM
        else:
            data = np.load(data_path)
            indices = data["indices"]
            labels = data["labels"]
            codebooks = data["codebooks"]
        return indices, labels, np.asarray(codebooks, dtype=np.float32)

    def setup(self, stage: str) -> None:
        """Datasets are already split in __init__."""
        pass

    def _get_dataloader(
        self, dataset: IterableDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Override for IterableDataset (no shuffle/drop_last; transforms is None)."""
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
