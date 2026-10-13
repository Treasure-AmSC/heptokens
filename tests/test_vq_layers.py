"""Dead-code revival in RevivingVectorQuantize / RevivingResidualVQ."""

import torch
from vector_quantize_pytorch import VectorQuantize

from heptokens.models.vq_layers import RevivingResidualVQ, RevivingVectorQuantize

DIM, K = 4, 8


def _pair(threshold):
    torch.manual_seed(0)
    base = VectorQuantize(dim=DIM, codebook_size=K)
    revive = RevivingVectorQuantize(dim=DIM, codebook_size=K, threshold_ema_dead_code=threshold)
    revive.load_state_dict(base.state_dict())
    return base, revive


def test_threshold_zero_matches_base_class():
    base, revive = _pair(threshold=0.0)
    x = torch.randn(2, 16, DIM)
    out_b, out_r = base(x), revive(x)
    for a, b in zip(out_b, out_r):
        assert torch.equal(a, b)
    for name, buf in base.state_dict().items():
        assert torch.equal(buf, revive.state_dict()[name])


def test_dead_codes_revived_to_batch_vectors_in_training():
    _, revive = _pair(threshold=1.0)
    x = torch.full((1, 64, DIM), 3.0)  # one distinct vector -> a single live code
    _, idx, _ = revive(x)
    live = int(idx.unique())
    dead = torch.arange(K) != live
    assert torch.equal(revive.embed[:, dead], torch.full((DIM, K - 1), 3.0))
    assert torch.equal(revive.embed_avg[:, dead], revive.embed[:, dead])
    assert torch.all(revive.cluster_size[dead] == 1.0)
    assert revive.cluster_size[live] >= 1.0


def test_no_update_in_eval_mode():
    _, revive = _pair(threshold=1.0)
    revive.eval()
    before = {k: v.clone() for k, v in revive.state_dict().items()}
    revive(torch.randn(1, 32, DIM))
    for name, buf in revive.state_dict().items():
        assert torch.equal(buf, before[name])


def test_residual_vq_output_is_sum_of_layer_quantizations():
    torch.manual_seed(0)
    rvq = RevivingResidualVQ(dim=DIM, codebook_size=K, num_quantizers=3, threshold_ema_dead_code=1.0)
    rvq.eval()
    x = torch.randn(2, 5, DIM)
    quantized, indices, losses = rvq(x)
    assert indices.shape == (3, 2, 5) and losses.shape == (3,)
    expected = sum(
        torch.nn.functional.embedding(indices[q], layer.embed.t()) for q, layer in enumerate(rvq.layers)
    )
    assert torch.allclose(quantized, expected, atol=1e-6)
