"""MemmapPredictionWriter rejects token indices that do not fit the int16 export format."""

import pytest
import torch

from heptokens.export_tokens import MemmapPredictionWriter


def _write(tmp_path, index: int) -> MemmapPredictionWriter:
    writer = MemmapPredictionWriter(str(tmp_path), total_jets=1, num_elements=2, num_quantizers=1)
    prediction = {"indices": torch.tensor([[[index], [-1]]]), "labels": torch.zeros(1)}
    writer.write_on_batch_end(None, None, prediction, None, {}, 0, 0)
    return writer


def test_largest_int16_index_is_written(tmp_path):
    writer = _write(tmp_path, 32767)
    assert writer.indices_mmap[0, :, 0].tolist() == [32767, -1]


def test_index_beyond_int16_raises(tmp_path):
    with pytest.raises(OverflowError):
        _write(tmp_path, 32768)
