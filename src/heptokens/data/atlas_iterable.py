"""For loading in jets from HDF files and creating a mappable dataset."""

import logging
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from heptokens.data.atlas_mappable import BaseMapModule
from heptokens.data.collation import collate_and_transform
from heptokens.utils.plot_physics import JET_FEATURES, TRACK_FEATURES

log = logging.getLogger(__name__)


def _iter_chunk_samples(
    jets_ds,
    tracks_ds,
    chunk_indices,
    *,
    jet_features,
    cst_features,
    label_key,
    num_csts,
    label_map,
    max_jet_pt,
    max_cst_pt,
    pt_idx,
    cst_pt_idx,
):
    """Read one chunk of (sorted) indices from open HDF5 datasets and yield samples.

    Applies the jet/constituent pT cuts and extracts the requested features, yielding
    per-sample dicts ``{jets, csts, mask, labels, eventNumber}``. Shared by
    :class:`IndexedIterMapDataset` and :class:`MultiIndexedIterDataset` so the
    read + cut + feature-extraction logic lives in exactly one place.
    """
    n = len(chunk_indices)

    # Use slice indexing when indices are contiguous (orders of magnitude faster).
    idx_start, idx_end = int(chunk_indices[0]), int(chunk_indices[-1]) + 1
    if idx_end - idx_start == n:
        sel = slice(idx_start, idx_end)
    else:
        sel = chunk_indices

    # Read the needed jet rows ONCE as a structured (compound) array, then slice
    # fields in memory. Per-field indexing (jets_ds[feat][sel]) on a gzip-compressed
    # compound dataset decompresses the ENTIRE column before subsetting, which is
    # ~180x slower on large files. A single jets_ds[sel] reads only the needed rows.
    rows = jets_ds[sel]

    # Load jet features for these indices
    jets_chunk = np.empty((n, len(jet_features)), dtype=np.float32)
    for i, feat in enumerate(jet_features):
        jets_chunk[:, i] = rows[feat]

    # Load labels
    labels_chunk = rows[label_key]
    labels_mapped = np.array([label_map[label] for label in labels_chunk])

    # Load event numbers
    event_numbers_chunk = rows["eventNumber"]

    # Load constituent features
    tracks_chunk = tracks_ds[sel, :num_csts]
    csts_chunk = np.empty((n, num_csts, len(cst_features)), dtype=np.float32)
    for i, key in enumerate(cst_features):
        csts_chunk[:, :, i] = tracks_chunk[key]

    # Load validity mask
    mask_chunk = tracks_chunk["valid"]

    # Apply jet pT cut
    keep = np.ones(n, dtype=bool)
    if max_jet_pt is not None and pt_idx is not None:
        keep &= jets_chunk[:, pt_idx] <= max_jet_pt
    # Apply constituent pT cut
    if max_cst_pt is not None and cst_pt_idx is not None:
        cst_pts = csts_chunk[:, :, cst_pt_idx]
        max_per_jet = np.where(mask_chunk, cst_pts, 0).max(axis=1)
        keep &= max_per_jet <= max_cst_pt

    # Yield individual samples
    for i in range(n):
        if not keep[i]:
            continue
        yield {
            "jets": jets_chunk[i],
            "csts": csts_chunk[i],
            "mask": mask_chunk[i],
            "labels": labels_mapped[i],
            "eventNumber": event_numbers_chunk[i],
        }


class IterMapDataset(IterableDataset):
    """Iterable dataset that loads jets on-demand without loading everything into memory."""

    def __init__(
        self,
        file_path: str,
        jet_features: list | None = None,
        cst_features: list | None = None,
        label_key: str = "HadronConeExclTruthLabelID",
        num_jets: int | None = None,
        num_csts: int | None = None,
        max_jet_pt: float | None = None,
        max_cst_pt: float | None = None,
        chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = TRACK_FEATURES

        self.file_path = file_path
        self.jet_features = jet_features
        self.cst_features = cst_features
        self.label_key = label_key
        self.num_csts = num_csts
        self.chunk_size = chunk_size
        self.max_jet_pt = max_jet_pt
        self.max_cst_pt = max_cst_pt
        self.pt_idx = jet_features.index("pt") if "pt" in jet_features else None
        self.cst_pt_idx = cst_features.index("pt") if "pt" in cst_features else None

        # Determine total number of jets and build label map
        with h5py.File(file_path, mode="r") as handle:
            jets_ds = handle["jets"]
            total_jets = len(jets_ds)
            self.num_jets = min(total_jets, num_jets) if num_jets is not None else total_jets

            # Load labels to build label map (labels are typically small)
            labels = jets_ds[self.label_key][: self.num_jets]
            unique_labels = np.unique(labels)
            self.label_map = {label: i for i, label in enumerate(unique_labels)}

            # Get number of constituents
            tracks_ds = handle["tracks"]
            total_csts = tracks_ds.shape[1]
            if num_csts is None:
                self.num_csts = total_csts
            else:
                self.num_csts = min(total_csts, num_csts)

        log.info(
            f"IterMapDataset: {self.num_jets} jets with {self.num_csts}"
            f"constituents from {self.file_path}"
        )

    def __iter__(self):
        """Iterate through dataset in chunks to manage memory."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            start_idx = 0
            end_idx = self.num_jets
        else:
            # Multi-process data loading: split workload among workers
            per_worker = int(np.ceil(self.num_jets / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.num_jets)

        # Open file once per worker
        with h5py.File(self.file_path, mode="r") as handle:
            jets_ds = handle["jets"]
            tracks_ds = handle["tracks"]

            # Process data in chunks
            for chunk_start in range(start_idx, end_idx, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, end_idx)

                # Read the needed jet rows ONCE as a structured (compound) array, then
                # slice fields in memory (see _iter_chunk_samples for rationale).
                rows = jets_ds[chunk_start:chunk_end]

                # Load jet features for this chunk
                jets_chunk = np.empty(
                    (chunk_end - chunk_start, len(self.jet_features)), dtype=np.float32
                )
                for i, feat in enumerate(self.jet_features):
                    jets_chunk[:, i] = rows[feat]

                # Load labels for this chunk
                labels_chunk = rows[self.label_key]
                labels_mapped = np.array([self.label_map[label] for label in labels_chunk])

                # Load event numbers for this chunk
                event_numbers_chunk = rows["eventNumber"]

                # Load constituent features for this chunk
                tracks_chunk = tracks_ds[chunk_start:chunk_end, : self.num_csts]
                csts_chunk = np.empty(
                    (chunk_end - chunk_start, self.num_csts, len(self.cst_features)),
                    dtype=np.float32,
                )
                for i, key in enumerate(self.cst_features):
                    csts_chunk[:, :, i] = tracks_chunk[key]

                # Load validity mask for this chunk
                mask_chunk = tracks_chunk["valid"]

                # Apply jet pT cut
                keep = np.ones(chunk_end - chunk_start, dtype=bool)
                if self.max_jet_pt is not None and self.pt_idx is not None:
                    keep &= jets_chunk[:, self.pt_idx] <= self.max_jet_pt
                # Apply constituent pT cut
                if self.max_cst_pt is not None and self.cst_pt_idx is not None:
                    cst_pts = csts_chunk[:, :, self.cst_pt_idx]
                    max_per_jet = np.where(mask_chunk, cst_pts, 0).max(axis=1)
                    keep &= max_per_jet <= self.max_cst_pt

                # Yield individual samples from this chunk
                for i in range(chunk_end - chunk_start):
                    if not keep[i]:
                        continue
                    yield {
                        "jets": jets_chunk[i],
                        "csts": csts_chunk[i],
                        "mask": mask_chunk[i],
                        "labels": labels_mapped[i],
                        "eventNumber": event_numbers_chunk[i],
                    }

    def __len__(self) -> int:
        """Return total number of jets."""
        return self.num_jets


class PresplitIterModule(BaseMapModule):
    """DataModule for pre-split train/val/test datasets using iterable datasets."""

    def __init__(
        self,
        *,
        train_path: str,
        val_path: str,
        test_path: str,
        chunk_size: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.chunk_size = chunk_size

        # Extract data_config from kwargs for dataset creation
        data_config = {k: v for k, v in self.data_config.items()}
        data_config["chunk_size"] = chunk_size

        # Initialize validation set early to get a sample
        self.valid_set = IterMapDataset(self.val_path, **data_config)

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        data_config = {k: v for k, v in self.data_config.items()}
        data_config["chunk_size"] = self.chunk_size

        if stage in {"fit", "train"}:
            self.train_set = IterMapDataset(self.train_path, **data_config)
        if stage in {"predict", "test"}:
            self.test_set = IterMapDataset(self.test_path, **data_config)

    def _get_dataloader(
        self, dataset: IterableDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Override dataloader creation for iterable datasets."""
        collate_fn = None
        if self.transforms is not None:
            collate_fn = partial(collate_and_transform, transforms=self.transforms)

        dataloader_kwargs = {}
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.persistent_workers
            if self.multiprocessing_context is not None:
                dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context

        # Note: shuffle parameter is ignored for IterableDataset
        # Shuffling would need to be implemented within the dataset's __iter__ method
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )


class SingleFileIterModule(BaseMapModule):
    """DataModule that splits a single dataset file into train/val/test using iterable datasets."""

    def __init__(
        self,
        *,
        data_path: str,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        chunk_size: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        self.data_path = data_path
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.chunk_size = chunk_size

        # Determine total size and split indices
        with h5py.File(data_path, mode="r") as handle:
            total_size = len(handle["jets"])

        # Calculate split sizes
        train_size = int(total_size * train_frac)
        val_size = int(total_size * val_frac)

        # Generate shuffled indices if seed is provided
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.indices = rng.permutation(total_size)
        else:
            self.indices = np.arange(total_size)

        # Split indices into train/val/test
        self.train_indices = self.indices[:train_size]
        self.val_indices = self.indices[train_size : train_size + val_size]
        self.test_indices = self.indices[train_size + val_size :]

        # Create dataset configs
        data_config = {k: v for k, v in self.data_config.items()}
        data_config["chunk_size"] = chunk_size

        # Initialize splits as IterMapDataset with index filtering
        self.train_set = IndexedIterMapDataset(
            self.data_path, indices=self.train_indices, **data_config
        )
        self.valid_set = IndexedIterMapDataset(
            self.data_path, indices=self.val_indices, **data_config
        )
        self.test_set = IndexedIterMapDataset(
            self.data_path, indices=self.test_indices, **data_config
        )

    def setup(self, stage: str) -> None:
        """Datasets are already split in __init__."""
        pass

    def _get_dataloader(
        self, dataset: IterableDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Override dataloader creation for iterable datasets."""
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


class IndexedIterMapDataset(IterableDataset):
    """Iterable dataset that loads specific indices from an HDF5 file."""

    def __init__(
        self,
        file_path: str,
        indices: np.ndarray,
        jet_features: list | None = None,
        cst_features: list | None = None,
        label_key: str = "HadronConeExclTruthLabelID",
        num_csts: int | None = None,
        max_jet_pt: float | None = None,
        max_cst_pt: float | None = None,
        chunk_size: int = 1000,
        label_map: dict | None = None,
    ) -> None:
        super().__init__()
        if jet_features is None:
            jet_features = JET_FEATURES
        if cst_features is None:
            cst_features = TRACK_FEATURES

        self.file_path = file_path
        self.indices = indices
        self.jet_features = jet_features
        self.cst_features = cst_features
        self.label_key = label_key
        self.chunk_size = chunk_size
        self.max_jet_pt = max_jet_pt
        self.max_cst_pt = max_cst_pt
        self.pt_idx = jet_features.index("pt") if "pt" in jet_features else None
        self.cst_pt_idx = cst_features.index("pt") if "pt" in cst_features else None

        # Build label map from all indices
        with h5py.File(file_path, mode="r") as handle:
            jets_ds = handle["jets"]

            # Use a provided (shared) label map when given; otherwise build one from this
            # file's indices. Existing callers pass no label_map and keep this behavior.
            if label_map is not None:
                self.label_map = label_map
            else:
                # Load labels for these indices (still memory efficient for most cases)
                labels = jets_ds[self.label_key][indices]
                unique_labels = np.unique(labels)
                self.label_map = {label: i for i, label in enumerate(unique_labels)}

            # Get number of constituents
            tracks_ds = handle["tracks"]
            total_csts = tracks_ds.shape[1]
            if num_csts is None:
                self.num_csts = total_csts
            else:
                self.num_csts = min(total_csts, num_csts)

        log.info(f"IndexedIterMapDataset: {len(indices)} jets with {self.num_csts} constituents")

    def __iter__(self):
        """Iterate through the specified indices in chunks."""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            worker_indices = self.indices
            initial_chunk = self.chunk_size
        else:
            # Multi-process data loading: split indices among workers
            per_worker = int(np.ceil(len(self.indices) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.indices))
            worker_indices = self.indices[start_idx:end_idx]
            # Stagger initial chunk size so workers don't all reload at the same time
            initial_chunk = max(
                1, self.chunk_size - worker_id * (self.chunk_size // max(worker_info.num_workers, 1))
            )

        # Sort indices for more efficient HDF5 access
        sorted_worker_indices = np.sort(worker_indices)

        # Open file once per worker
        with h5py.File(self.file_path, mode="r") as handle:
            jets_ds = handle["jets"]
            tracks_ds = handle["tracks"]

            # Process in chunks (first chunk may be smaller to stagger workers)
            pos = 0
            first = True
            while pos < len(sorted_worker_indices):
                cs = initial_chunk if first else self.chunk_size
                first = False
                chunk_end = min(pos + cs, len(sorted_worker_indices))
                chunk_indices = sorted_worker_indices[pos:chunk_end]
                pos = chunk_end

                yield from _iter_chunk_samples(
                    jets_ds,
                    tracks_ds,
                    chunk_indices,
                    jet_features=self.jet_features,
                    cst_features=self.cst_features,
                    label_key=self.label_key,
                    num_csts=self.num_csts,
                    label_map=self.label_map,
                    max_jet_pt=self.max_jet_pt,
                    max_cst_pt=self.max_cst_pt,
                    pt_idx=self.pt_idx,
                    cst_pt_idx=self.cst_pt_idx,
                )

    def __len__(self) -> int:
        """Return number of indices."""
        return len(self.indices)


class MultiIndexedIterDataset(IterableDataset):
    """Iterable dataset over several HDF5 files, interleaving blocks across files.

    Raw-feature counterpart to :class:`heptokens.data.token_npz.TokenNpzDataset`. Holds a
    list of *segments* ``[{"file_path": str, "split_indices": np.ndarray}, ...]`` (one per
    input file for a given split). In ``__iter__`` it builds a flat list of
    ``(segment_id, chunk_start)`` blocks across all segments, shards whole blocks across
    workers deterministically, optionally shuffles the block order per epoch, reads each
    block with the same slice-vs-fancy-index optimization and cut logic as
    :class:`IndexedIterMapDataset` (via the shared ``_iter_chunk_samples`` helper), and
    emits per-sample dicts through an optional streaming shuffle buffer.
    """

    def __init__(
        self,
        segments: list[dict],
        *,
        jet_features: list,
        cst_features: list,
        label_key: str,
        num_csts: int,
        label_map: dict,
        max_jet_pt: float | None = None,
        max_cst_pt: float | None = None,
        chunk_size: int = 1000,
        shuffle: bool = False,
        shuffle_buffer: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not segments:
            raise ValueError("MultiIndexedIterDataset requires at least one segment")
        self.segments = segments
        self.jet_features = jet_features
        self.cst_features = cst_features
        self.label_key = label_key
        self.num_csts = num_csts
        self.label_map = label_map
        self.max_jet_pt = max_jet_pt
        self.max_cst_pt = max_cst_pt
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self._epoch = 0
        self.pt_idx = jet_features.index("pt") if "pt" in jet_features else None
        self.cst_pt_idx = cst_features.index("pt") if "pt" in cst_features else None
        self._length = int(sum(len(s["split_indices"]) for s in segments))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # Per-epoch RNG seeded from (seed, worker_id, epoch): reproducible split, varying
        # shuffle order across epochs (mirrors TokenNpzDataset).
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

        # Lazily open + cache one file handle (and its datasets) per file for this worker.
        handles: dict[str, tuple] = {}
        buffer: list[dict] = []
        try:
            for sid, pos in blocks:
                si = seg_sorted[sid]
                file_path = self.segments[sid]["file_path"]
                if file_path not in handles:
                    fh = h5py.File(file_path, mode="r")
                    handles[file_path] = (fh, fh["jets"], fh["tracks"])
                _, jets_ds, tracks_ds = handles[file_path]
                chunk_indices = si[pos : pos + self.chunk_size]

                for sample in _iter_chunk_samples(
                    jets_ds,
                    tracks_ds,
                    chunk_indices,
                    jet_features=self.jet_features,
                    cst_features=self.cst_features,
                    label_key=self.label_key,
                    num_csts=self.num_csts,
                    label_map=self.label_map,
                    max_jet_pt=self.max_jet_pt,
                    max_cst_pt=self.max_cst_pt,
                    pt_idx=self.pt_idx,
                    cst_pt_idx=self.cst_pt_idx,
                ):
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
        finally:
            for fh, _, _ in handles.values():
                fh.close()

    def __len__(self) -> int:
        return self._length


class MultiFileIterModule(BaseMapModule):
    """DataModule that splits and combines several HDF5 files into one streamed dataset.

    Raw-feature counterpart to :class:`heptokens.data.token_npz.TokenNpzModule`: each file
    is split independently with the same seeded fractions (so the combined ratios are
    preserved), a single SHARED label map is used across all files, and the per-split
    segments are interleaved + shuffle-buffered into one :class:`MultiIndexedIterDataset`
    stream. Use this to train on the union of the small+medium+large h5 files (~200M jets),
    analogous to ``token_clf_full`` over the three token ``.npz`` files.
    """

    def __init__(
        self,
        *,
        data_paths: list[str],
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        chunk_size: int = 1000,
        shuffle_buffer: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not data_paths:
            raise ValueError("MultiFileIterModule requires a non-empty data_paths list")
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        self.data_paths = list(data_paths)
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer

        # data_config carries jet_features/cst_features/label_key/num_csts/max_*_pt.
        cfg = dict(self.data_config)
        self.jet_features = cfg.get("jet_features") or JET_FEATURES
        self.cst_features = cfg.get("cst_features") or TRACK_FEATURES
        self.label_key = cfg.get("label_key", "HadronConeExclTruthLabelID")
        self.max_jet_pt = cfg.get("max_jet_pt")
        self.max_cst_pt = cfg.get("max_cst_pt")
        num_csts_cfg = cfg.get("num_csts")

        # Build a single SHARED label map from the union of unique labels across all files
        # (sorted), so every file maps the same physics label to the same class index. For
        # ATLAS HadronConeExclTruthLabelID {0,4,5,15} this yields {0:0,4:1,5:2,15:3}
        # (light, c, b, tau). This avoids per-file maps disagreeing when a file is missing
        # a rare flavor.
        unique_union: set = set()
        per_file_counts: list[int] = []
        total_csts: int | None = None
        for path in self.data_paths:
            with h5py.File(path, mode="r") as handle:
                file_unique = np.unique(handle["jets"][self.label_key])
                unique_union.update(file_unique.tolist())
                per_file_counts.append(len(handle["jets"]))
                file_csts = handle["tracks"].shape[1]
                total_csts = file_csts if total_csts is None else min(total_csts, file_csts)
        sorted_labels = sorted(unique_union)
        self.label_map = {label: i for i, label in enumerate(sorted_labels)}
        self.num_csts = total_csts if num_csts_cfg is None else min(total_csts, num_csts_cfg)
        log.info(
            f"MultiFileIterModule: {len(self.data_paths)} files, "
            f"{sum(per_file_counts):,} jets total, shared label_map={self.label_map}, "
            f"num_csts={self.num_csts}"
        )

        # Split each file independently with the same fractions/seed so combined ratios
        # are preserved (mirrors TokenNpzModule).
        train_segs: list[dict] = []
        val_segs: list[dict] = []
        test_segs: list[dict] = []
        for path, n in zip(self.data_paths, per_file_counts):
            train_size = int(n * train_frac)
            val_size = int(n * val_frac)
            if seed is not None:
                perm = np.random.default_rng(seed).permutation(n)
            else:
                perm = np.arange(n)
            train_idx = perm[:train_size]
            val_idx = perm[train_size : train_size + val_size]
            test_idx = perm[train_size + val_size :]
            train_segs.append({"file_path": path, "split_indices": train_idx})
            val_segs.append({"file_path": path, "split_indices": val_idx})
            test_segs.append({"file_path": path, "split_indices": test_idx})

        ds_kwargs = dict(
            jet_features=self.jet_features,
            cst_features=self.cst_features,
            label_key=self.label_key,
            num_csts=self.num_csts,
            label_map=self.label_map,
            max_jet_pt=self.max_jet_pt,
            max_cst_pt=self.max_cst_pt,
            chunk_size=chunk_size,
        )
        # Train: shuffled block order + streaming shuffle buffer. Val/test: file order.
        self.train_set = MultiIndexedIterDataset(
            train_segs, shuffle=True, shuffle_buffer=shuffle_buffer, seed=seed, **ds_kwargs
        )
        self.valid_set = MultiIndexedIterDataset(val_segs, **ds_kwargs)
        self.test_set = MultiIndexedIterDataset(test_segs, **ds_kwargs)

    def setup(self, stage: str) -> None:
        """Datasets are already built in __init__."""
        pass

    def _get_dataloader(
        self, dataset: IterableDataset, shuffle: bool, drop_last: bool
    ) -> DataLoader:
        """Override dataloader creation for iterable datasets (shuffle/drop_last ignored)."""
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
