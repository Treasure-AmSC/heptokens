"""Focused tests for optional, export-layer-only position tokenization.

These exercise ``MemmapPredictionWriter`` directly (bypassing
``trainer.predict()``/``LitVqVae.predict_step``), so they do not require
training a model and do not depend on the pre-existing ``batch["labels"]``
requirement of ``predict_step`` (the writer only needs a "labels" entry in
the synthetic ``prediction`` dict, which these tests supply directly).

Key invariant under test: position tokenization is entirely opt-in and lives
in the export layer. ``LitVqVae``/``predict_step`` are not touched by this
feature at all — the writer reads ``positions``/mask directly from the raw
dataloader ``batch`` argument that Lightning already passes to
``write_on_batch_end``.
"""

from pathlib import Path

import numpy as np
import torch

from heptokens.export_tokens import MemmapPredictionWriter

TOTAL = 4
NUM_ELEMENTS = 5
NUM_QUANTIZERS = 2
NUM_POS_FEATURES = 3


class _FakePositionTokenizer:
    """Minimal duck-typed stand-in for heptokens_cocoa's PositionTokenizer.

    Any object satisfying this contract (callable -> LongTensor[..., n_features]
    with attribute n_features) works with MemmapPredictionWriter.
    """

    n_features = NUM_POS_FEATURES

    def __call__(self, positions: torch.Tensor) -> torch.Tensor:
        # Simple deterministic non-learned binning for test purposes.
        return torch.clamp((positions * 10).long(), min=0, max=99)


def _synthetic_prediction_and_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    indices = torch.randint(0, 8, (TOTAL, NUM_ELEMENTS, NUM_QUANTIZERS), generator=g)
    labels = torch.randint(0, 3, (TOTAL,), generator=g)
    mask = torch.ones(TOTAL, NUM_ELEMENTS, dtype=torch.bool)
    mask[0, -2:] = False  # a couple of masked (padded) elements

    prediction = {"indices": indices, "labels": labels}
    batch = {
        "positions": torch.rand(TOTAL, NUM_ELEMENTS, NUM_POS_FEATURES, generator=g),
        "mask": mask,
    }
    return prediction, batch


def test_no_pos_tokenizer_omits_pos_tokens(tmp_path: Path):
    prediction, batch = _synthetic_prediction_and_batch()

    writer = MemmapPredictionWriter(
        str(tmp_path), TOTAL, NUM_ELEMENTS, NUM_QUANTIZERS
    )  # pos_tokenizer not provided -> default None
    writer.write_on_batch_end(None, None, prediction, None, batch, 0, 0)
    n_written = writer.finalize()

    assert writer.has_pos_tokens is False
    assert writer.pos_tokens_path is None
    assert not (tmp_path / "pos_tokens.npy").exists()

    indices = np.lib.format.open_memmap(str(writer.indices_path), mode="r")[:n_written]
    labels = np.lib.format.open_memmap(str(writer.labels_path), mode="r")[:n_written]
    assert np.array_equal(indices, prediction["indices"].numpy().astype(np.int16))
    assert np.array_equal(labels, prediction["labels"].numpy().astype(np.int8))


def test_pos_tokenizer_adds_pos_tokens_without_changing_indices(tmp_path: Path):
    prediction, batch = _synthetic_prediction_and_batch()
    tok = _FakePositionTokenizer()

    # Baseline run with no tokenizer, for comparison.
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_writer = MemmapPredictionWriter(
        str(baseline_dir), TOTAL, NUM_ELEMENTS, NUM_QUANTIZERS
    )
    baseline_writer.write_on_batch_end(None, None, prediction, None, batch, 0, 0)
    n_baseline = baseline_writer.finalize()
    baseline_indices = np.lib.format.open_memmap(str(baseline_writer.indices_path), mode="r")[
        :n_baseline
    ]
    baseline_labels = np.lib.format.open_memmap(str(baseline_writer.labels_path), mode="r")[
        :n_baseline
    ]

    # Run with the tokenizer enabled.
    tok_dir = tmp_path / "with_tokenizer"
    tok_dir.mkdir()
    writer = MemmapPredictionWriter(
        str(tok_dir),
        TOTAL,
        NUM_ELEMENTS,
        NUM_QUANTIZERS,
        pos_tokenizer=tok,
        num_pos_features=tok.n_features,
    )
    writer.write_on_batch_end(None, None, prediction, None, batch, 0, 0)
    n_written = writer.finalize()

    assert writer.has_pos_tokens is True
    assert writer.pos_tokens_path is not None
    assert (tok_dir / "pos_tokens.npy").exists()

    indices = np.lib.format.open_memmap(str(writer.indices_path), mode="r")[:n_written]
    labels = np.lib.format.open_memmap(str(writer.labels_path), mode="r")[:n_written]
    pos_tokens = np.lib.format.open_memmap(str(writer.pos_tokens_path), mode="r")[:n_written]

    # Enabling pos tokenization must not perturb VQ indices/labels at all.
    assert np.array_equal(indices, baseline_indices)
    assert np.array_equal(labels, baseline_labels)

    assert pos_tokens.shape == (TOTAL, NUM_ELEMENTS, NUM_POS_FEATURES)
    assert pos_tokens.dtype == np.int16

    expected = tok(batch["positions"]).numpy().astype(np.int16)
    mask = batch["mask"].numpy()
    expected[~mask] = -1
    assert np.array_equal(pos_tokens, expected)
    # masked elements are -1
    assert (pos_tokens[0, -2:] == -1).all()
