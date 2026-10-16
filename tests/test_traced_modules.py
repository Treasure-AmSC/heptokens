"""Unit tests for heptokens.traced_modules — TDD, one behavior per cycle."""

from functools import partial

import pytest
import torch
import torch.nn as nn

from heptokens.models.coders import BaseDecoder, CoderModel, Decoder, Encoder
from heptokens.models.vq_vae import LitVqVae
from heptokens.traced_modules import (
    CodebookLookup,
    TensorDecoder,
    TensorEncoder,
    TensorIndexEncoder,
    VqVaeKit,
    build_inference_modules,
    save_inference_modules,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Q, K, D = 2, 8, 4  # num_quantizers, codebook_size, codebook_dim
B, T = 3, 5  # batch, tokens


@pytest.fixture
def codebooks():
    torch.manual_seed(0)
    return torch.randn(Q, K, D)


@pytest.fixture
def indices_valid():
    """All-valid indices in [0, K)."""
    torch.manual_seed(1)
    return torch.randint(0, K, (B, T, Q))


# ---------------------------------------------------------------------------
# CodebookLookup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TensorIndexEncoder
# ---------------------------------------------------------------------------


def _make_index_encoder(simple_mlp, codebooks):
    """Helper: build a TensorIndexEncoder from fixtures."""
    tensor_enc = TensorEncoder(simple_mlp)
    embed_list = nn.ParameterList(
        [nn.Parameter(codebooks[q].t().clone(), requires_grad=False) for q in range(Q)]
    )  # each [D, K]
    return TensorIndexEncoder(tensor_enc, embed_list)


def test_index_encoder_output_shape(simple_mlp, codebooks, valid_mask):
    idx_enc = _make_index_encoder(simple_mlp, codebooks)
    x = torch.randn(B, T, 3)
    indices = idx_enc(x, valid_mask)
    assert indices.shape == (B, T, Q)


def test_index_encoder_indices_in_range(simple_mlp, codebooks, valid_mask):
    idx_enc = _make_index_encoder(simple_mlp, codebooks)
    x = torch.randn(B, T, 3)
    indices = idx_enc(x, valid_mask)
    valid_pos = valid_mask.unsqueeze(-1).expand_as(indices)
    assert (indices[valid_pos] >= 0).all()
    assert (indices[valid_pos] < K).all()


def test_index_encoder_masked_positions_are_neg1(simple_mlp, codebooks, valid_mask):
    idx_enc = _make_index_encoder(simple_mlp, codebooks)
    x = torch.randn(B, T, 3)
    indices = idx_enc(x, valid_mask)
    # valid_mask[0, -2:] is False
    assert (indices[0, -2:] == -1).all()


def test_index_encoder_survives_script_and_roundtrip(simple_mlp, codebooks, valid_mask, tmp_path):
    idx_enc = _make_index_encoder(simple_mlp, codebooks)
    scripted = torch.jit.script(idx_enc)
    x = torch.randn(B, T, 3)
    expected = scripted(x, valid_mask)
    path = tmp_path / "index_encoder.pt"
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path))
    assert torch.equal(loaded(x, valid_mask), expected)


def test_codebook_lookup_output_shape(codebooks, indices_valid):
    lkp = CodebookLookup(codebooks)
    z_q = lkp(indices_valid)
    assert z_q.shape == (B, T, D)


def test_codebook_lookup_sums_residuals(codebooks, indices_valid):
    """Each output vector = sum of codebook[q, idx_q] across quantizers."""
    lkp = CodebookLookup(codebooks)
    z_q = lkp(indices_valid)

    # Manually compute expected sum for first element
    expected = torch.zeros(D)
    for q in range(Q):
        expected = expected + codebooks[q, indices_valid[0, 0, q]]
    assert torch.allclose(z_q[0, 0], expected, atol=1e-6)


def test_codebook_lookup_masked_positions_are_zero(codebooks):
    """Positions with index -1 produce a zero output vector."""
    lkp = CodebookLookup(codebooks)
    indices = torch.full((B, T, Q), -1, dtype=torch.long)  # all masked
    z_q = lkp(indices)
    assert torch.all(z_q == 0)


def test_codebook_lookup_mixed_mask(codebooks):
    """Only fully masked (all-quantizer -1) positions are zeroed."""
    lkp = CodebookLookup(codebooks)
    torch.manual_seed(2)
    indices = torch.randint(0, K, (B, T, Q))
    indices[0, 0, :] = -1  # mask first position entirely

    z_q = lkp(indices)
    assert torch.all(z_q[0, 0] == 0), "masked position should be zero"
    assert not torch.all(z_q[0, 1] == 0), "valid position should be nonzero"


@pytest.fixture
def simple_mlp():
    """Tiny MLP: D_in=3 -> hidden=8 -> D_out=4 (codebook_dim)."""
    return nn.Sequential(nn.Linear(3, 8), nn.LayerNorm(8), nn.ReLU(), nn.Linear(8, 4))


@pytest.fixture
def valid_mask():
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -2:] = False  # last 2 positions of first sample are masked
    return mask


# ---------------------------------------------------------------------------
# TensorEncoder
# ---------------------------------------------------------------------------


def test_encoder_output_shape(simple_mlp, valid_mask):
    enc = TensorEncoder(simple_mlp)
    x = torch.randn(B, T, 3)
    z_e = enc(x, valid_mask)
    assert z_e.shape == (B, T, 4)


def test_encoder_masked_positions_are_zero(simple_mlp, valid_mask):
    enc = TensorEncoder(simple_mlp)
    x = torch.randn(B, T, 3)
    z_e = enc(x, valid_mask)
    assert torch.all(z_e[0, -2:] == 0), "masked positions should be zero"
    assert not torch.all(z_e[0, 0] == 0), "valid position should be nonzero"


def test_encoder_survives_script_and_roundtrip(simple_mlp, valid_mask, tmp_path):
    enc = TensorEncoder(simple_mlp)
    scripted = torch.jit.script(enc)
    x = torch.randn(B, T, 3)
    expected = scripted(x, valid_mask)
    path = tmp_path / "encoder.pt"
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path))
    assert torch.allclose(loaded(x, valid_mask), expected)


# ---------------------------------------------------------------------------
# TensorDecoder
# ---------------------------------------------------------------------------


def test_decoder_output_shape(simple_mlp, valid_mask):
    # Decoder MLP: codebook_dim=4 -> hidden -> D_in=3
    dec_mlp = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.ReLU(), nn.Linear(8, 3))
    dec = TensorDecoder(dec_mlp)
    z_q = torch.randn(B, T, 4)
    x_hat = dec(z_q, valid_mask)
    assert x_hat.shape == (B, T, 3)


def test_decoder_masked_positions_are_zero(valid_mask):
    dec_mlp = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.ReLU(), nn.Linear(8, 3))
    dec = TensorDecoder(dec_mlp)
    z_q = torch.randn(B, T, 4)
    x_hat = dec(z_q, valid_mask)
    assert torch.all(x_hat[0, -2:] == 0), "masked positions should be zero"


def test_decoder_survives_script_and_roundtrip(valid_mask, tmp_path):
    dec_mlp = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.ReLU(), nn.Linear(8, 3))
    dec = TensorDecoder(dec_mlp)
    scripted = torch.jit.script(dec)
    z_q = torch.randn(B, T, 4)
    expected = scripted(z_q, valid_mask)
    path = tmp_path / "decoder.pt"
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path))
    assert torch.allclose(loaded(z_q, valid_mask), expected)


def test_codebook_lookup_survives_script_and_roundtrip(codebooks, indices_valid, tmp_path):
    """Module scripts cleanly and survives save/load without heptokens."""
    lkp = CodebookLookup(codebooks)
    scripted = torch.jit.script(lkp)
    expected = scripted(indices_valid)

    path = tmp_path / "codebook_lookup.pt"
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path))
    assert torch.allclose(loaded(indices_valid), expected)


# ---------------------------------------------------------------------------
# build_inference_modules / save_inference_modules / VqVaeKit
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_vqvae():
    """Minimal LitVqVae (codebook_dim=4, codebook_size=8, Q=2, input_dim=3)."""
    data_sample = torch.randn(1, T, 3)
    model = LitVqVae(
        encoder=partial(Encoder, model=partial(CoderModel, hidden_dims=[8, 16])),
        decoder=partial(Decoder, model=partial(CoderModel, hidden_dims=[16, 8])),
        codebook_size=K,
        codebook_dim=D,
        num_quantizers=Q,
        data_sample={"csts": data_sample, "mask": torch.ones(1, T, dtype=torch.bool)},
    )
    model.eval()
    return model


def test_build_inference_modules_returns_all_keys(tiny_vqvae):
    modules = build_inference_modules(tiny_vqvae)
    assert set(modules.keys()) == {"encoder", "decoder", "codebook_lookup", "index_encoder"}


def test_build_inference_modules_all_scripted(tiny_vqvae):
    modules = build_inference_modules(tiny_vqvae)
    for name, mod in modules.items():
        assert isinstance(mod, torch.jit.ScriptModule), f"{name} is not a ScriptModule"


def test_build_inference_modules_raises_for_unsupported_decoder(tiny_vqvae):
    class CustomDecoder(BaseDecoder):
        def forward(self, z_q, batch):
            return z_q

        def compute_loss(self, z_q, batch):
            return z_q.sum()

    tiny_vqvae.decoder = CustomDecoder()
    with pytest.raises(NotImplementedError):
        build_inference_modules(tiny_vqvae)


def test_save_inference_modules_creates_files(tiny_vqvae, tmp_path):
    modules_dir = save_inference_modules(tiny_vqvae, tmp_path)
    for name in ("encoder.pt", "decoder.pt", "codebook_lookup.pt", "index_encoder.pt"):
        assert (modules_dir / name).exists(), f"{name} not found"


def test_vqvaekit_load_and_roundtrip(tiny_vqvae, tmp_path):
    save_inference_modules(tiny_vqvae, tmp_path)
    kit = VqVaeKit.load(tmp_path / "modules")

    x = torch.randn(B, T, 3)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1] = False

    z_e = kit.encode_float(x, mask)
    assert z_e.shape == (B, T, D)
    assert torch.all(z_e[0, -1] == 0), "masked position should be zero in z_e"

    indices = kit.encode_indices(x, mask)
    assert indices.shape == (B, T, Q)
    assert (indices[0, -1] == -1).all()

    z_q = kit.lookup(indices)
    assert z_q.shape == (B, T, D)

    x_hat = kit.decode(z_q, mask)
    assert x_hat.shape == (B, T, 3)

    x_hat2 = kit.decode_from_indices(indices, mask)
    assert torch.allclose(x_hat, x_hat2)
