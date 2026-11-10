"""Sum pooling in SetToVectorTransformer keeps the scaling with token multiplicity."""

import torch

from heptokens.models.transformer import SetToVectorTransformer


def test_sum_pooling_scales_with_duplicated_tokens():
    torch.manual_seed(0)
    model = SetToVectorTransformer(
        input_dim=3, output_dim=2, d_model=16, n_heads=2, num_layers=1, pooling="sum"
    ).eval()
    x = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    x2, mask2 = torch.cat([x, x], dim=1), torch.cat([mask, mask], dim=1)

    with torch.no_grad():
        out, out2 = model(x, mask), model(x2, mask2)

    # Duplicating every token leaves each token's attention output unchanged, so the pooled
    # sum doubles; through the affine output head that is 2 * out - bias.
    bias = model.output_proj.bias
    assert torch.allclose(out2, 2 * out - bias, atol=1e-5)
