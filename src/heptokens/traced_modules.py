"""Scriptable inference modules for VQ-VAE — usable without heptokens installed.

Saved to <output_dir>/modules/ by export_tokens.py:
  encoder.pt           TensorEncoder        (x, mask) -> z_e
  decoder.pt           TensorDecoder        (z_q, mask) -> x_hat
  codebook_lookup.pt   CodebookLookup       (indices)  -> z_q
  index_encoder.pt     TensorIndexEncoder   (x, mask)  -> indices

Without heptokens:
    enc = torch.jit.load("modules/encoder.pt")
    z_e = enc(x, mask)                         # input -> pre-quantization float
    idx = torch.jit.load("modules/index_encoder.pt")
    indices = idx(x, mask)                     # input -> int tokens [B,T,Q]
    lkp = torch.jit.load("modules/codebook_lookup.pt")
    z_q = lkp(indices)                         # tokens -> summed float embedding
    dec = torch.jit.load("modules/decoder.pt")
    x_hat = dec(z_q, mask)                     # float embedding -> input space

With heptokens installed:
    from heptokens.traced_modules import VqVaeKit
    kit = VqVaeKit.load("modules/")
    x_hat = kit.decode_from_indices(indices, mask)
"""

import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from heptokens.models.vq_vae import LitVqVae

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scriptable modules
# ---------------------------------------------------------------------------


class CodebookLookup(nn.Module):
    """Look up residual-VQ codebook entries and sum them.

    Args:
        codebooks: ``[num_quantizers, codebook_size, codebook_dim]`` float tensor.
            This is the same array stored in the exported ``.npz`` under the
            ``codebooks`` key, so users can also reconstruct this module
            directly from the saved numpy array.

    Usage (no heptokens needed)::

        lkp = torch.jit.load("modules/codebook_lookup.pt")
        z_q = lkp(indices)   # indices: [B, T, Q] int — -1 = masked
    """

    num_quantizers: int

    def __init__(self, codebooks: torch.Tensor) -> None:
        super().__init__()
        # codebooks: [Q, K, D]
        self.register_buffer("codebooks", codebooks.float())
        self.num_quantizers = codebooks.shape[0]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Sum codebook embeddings across quantizers.

        Args:
            indices: ``[..., Q]`` int tensor; ``-1`` marks masked positions.

        Returns:
            ``[..., codebook_dim]`` float tensor; masked positions are zero.
        """
        valid = indices >= 0  # [..., Q] bool
        safe = indices.clamp(min=0).long()

        # Accumulate residual embeddings
        z_q = F.embedding(safe[..., 0], self.codebooks[0])  # [..., D]
        for q in range(1, self.num_quantizers):
            z_q = z_q + F.embedding(safe[..., q], self.codebooks[q])

        # Zero out positions masked in any quantizer
        all_valid = valid.all(dim=-1, keepdim=True).to(z_q.dtype)  # [..., 1]
        return z_q * all_valid


class TensorEncoder(nn.Module):
    """Apply the encoder MLP to all token positions.

    Wraps the ``nn.Sequential`` extracted from the trained ``Encoder``.
    Applies the MLP to **all** positions (including masked ones) for
    trace-safety, then zeros masked positions in the output.

    Args:
        mlp: The ``encoder.coder.model`` sequential from a trained VQ-VAE.

    Usage::

        enc = torch.jit.load("modules/encoder.pt")
        z_e = enc(x, mask)   # x: [B,T,D_in], mask: [B,T] bool -> [B,T,codebook_dim]
    """

    def __init__(self, mlp: nn.Sequential) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode inputs to pre-quantization float embeddings.

        Args:
            x: ``[B, T, D_in]`` float input tensor.
            mask: ``[B, T]`` bool tensor; ``True`` = valid token.

        Returns:
            ``[B, T, codebook_dim]`` float; masked positions are zero.
        """
        B, T = x.shape[0], x.shape[1]
        z = self.mlp(x.reshape(B * T, x.shape[2])).reshape(B, T, -1)
        return z * mask.unsqueeze(-1).to(z.dtype)


class TensorDecoder(nn.Module):
    """Apply the decoder MLP to all token positions.

    Wraps the ``nn.Sequential`` extracted from the trained ``Decoder``.

    Args:
        mlp: The ``decoder.coder.model`` sequential from a trained VQ-VAE.

    Usage::

        dec = torch.jit.load("modules/decoder.pt")
        x_hat = dec(z_q, mask)   # z_q: [B,T,D], mask: [B,T] bool -> [B,T,D_in]
    """

    def __init__(self, mlp: nn.Sequential) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode quantized embeddings to input space.

        Args:
            z_q: ``[B, T, codebook_dim]`` float tensor.
            mask: ``[B, T]`` bool tensor; ``True`` = valid token.

        Returns:
            ``[B, T, D_in]`` float; masked positions are zero.
        """
        B, T = z_q.shape[0], z_q.shape[1]
        x_hat = self.mlp(z_q.reshape(B * T, z_q.shape[2])).reshape(B, T, -1)
        return x_hat * mask.unsqueeze(-1).to(x_hat.dtype)


class TensorIndexEncoder(nn.Module):
    """Encode inputs all the way to integer codebook indices.

    Combines the encoder MLP with residual-VQ argmin quantization.
    Stores codebook embed matrices as frozen parameters so the module is
    self-contained after scripting.

    Args:
        encoder: A :class:`TensorEncoder` wrapping the trained encoder MLP.
        embed_list: ``nn.ParameterList`` of ``[codebook_dim, codebook_size]``
            tensors — one per quantizer (same layout as ``layer.embed`` in
            ``vector_quantize_pytorch``).

    Usage::

        idx = torch.jit.load("modules/index_encoder.pt")
        indices = idx(x, mask)   # [B,T,D_in], [B,T] bool -> [B,T,Q] int64
        # -1 at masked positions
    """

    def __init__(self, encoder: TensorEncoder, embed_list: nn.ParameterList) -> None:
        super().__init__()
        self.encoder = encoder
        self.embeds = embed_list  # each [D, K]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode inputs to integer indices via argmin residual quantization.

        Args:
            x: ``[B, T, D_in]`` float input tensor.
            mask: ``[B, T]`` bool tensor; ``True`` = valid token.

        Returns:
            ``[B, T, num_quantizers]`` int64 tensor; ``-1`` at masked positions.
        """
        z_e = self.encoder(x, mask)  # [B, T, D]
        B, T = x.shape[0], x.shape[1]
        flat = z_e.reshape(B * T, z_e.shape[2])  # [B*T, D]
        residual = flat.clone()

        all_indices: List[torch.Tensor] = []
        for embed in self.embeds:  # embed: [D, K]
            dist = (
                residual.pow(2).sum(1, keepdim=True)  # [N, 1]
                - 2.0 * residual @ embed  # [N, K]
                + embed.pow(2).sum(0, keepdim=True)  # [1, K]
            )
            idx = (-dist).max(1).indices  # [N] int64
            quantized = F.embedding(idx, embed.t())  # [N, D]
            residual = residual - quantized
            all_indices.append(idx.reshape(B, T))

        indices = torch.stack(all_indices, dim=-1)  # [B, T, Q]
        return indices.masked_fill(~mask.unsqueeze(-1), -1)


# ---------------------------------------------------------------------------
# Build and save helpers
# ---------------------------------------------------------------------------


def build_inference_modules(model: LitVqVae) -> Dict[str, nn.Module]:
    """Build the four scriptable inference modules from a trained VQ-VAE.

    Supports the standard ``Encoder`` / ``Decoder`` (``Coder``-based, with a
    ``.coder.model`` attribute that is an ``nn.Sequential``).  Custom encoder
    or decoder architectures that don't expose ``.coder.model`` will raise
    ``NotImplementedError``.

    Args:
        model: A loaded :class:`~heptokens.models.vq_vae.LitVqVae` instance.

    Returns:
        Dict with keys ``"encoder"``, ``"decoder"``, ``"codebook_lookup"``,
        ``"index_encoder"`` — all scripted via ``torch.jit.script``.

    Raises:
        NotImplementedError: If the encoder or decoder architecture is not
            the standard ``Coder``-based MLP type.
    """
    encoder = model.encoder
    decoder = model.decoder

    if not (hasattr(encoder, "coder") and hasattr(encoder.coder, "model")):
        raise NotImplementedError(
            f"Cannot export traced modules: encoder type {type(encoder).__name__!r} "
            "does not expose '.coder.model'. Only the default Coder-based MLP "
            "encoder is supported."
        )
    if not (hasattr(decoder, "coder") and hasattr(decoder.coder, "model")):
        raise NotImplementedError(
            f"Cannot export traced modules: decoder type {type(decoder).__name__!r} "
            "does not expose '.coder.model'. Only the default Coder-based MLP "
            "decoder is supported (DiffusionDecoder is not yet supported)."
        )

    tensor_encoder = TensorEncoder(encoder.coder.model)
    tensor_decoder = TensorDecoder(decoder.coder.model)

    # codebooks: [Q, K, D] from extract_codebooks-compatible layout
    vq = model.vector_quantization
    codebooks = torch.stack([layer.embed.t().detach() for layer in vq.layers])  # [Q, K, D]
    codebook_lookup = CodebookLookup(codebooks)

    # embed_list: [D, K] per quantizer (same as layer.embed)
    embed_list = nn.ParameterList(
        [nn.Parameter(layer.embed.detach().clone(), requires_grad=False) for layer in vq.layers]
    )
    index_encoder = TensorIndexEncoder(tensor_encoder, embed_list)

    return {
        "encoder": torch.jit.script(tensor_encoder),
        "decoder": torch.jit.script(tensor_decoder),
        "codebook_lookup": torch.jit.script(codebook_lookup),
        "index_encoder": torch.jit.script(index_encoder),
    }


def save_inference_modules(model: LitVqVae, output_dir: Path) -> Path:
    """Build, script, and save the four inference modules to ``output_dir/modules/``.

    Args:
        model: A loaded :class:`~heptokens.models.vq_vae.LitVqVae` instance.
        output_dir: Directory where the ``modules/`` subdirectory will be created.

    Returns:
        Path to the ``modules/`` directory.
    """
    output_dir = Path(output_dir)
    modules_dir = output_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)

    modules = build_inference_modules(model)
    names = {
        "encoder": "encoder.pt",
        "decoder": "decoder.pt",
        "codebook_lookup": "codebook_lookup.pt",
        "index_encoder": "index_encoder.pt",
    }
    for key, filename in names.items():
        torch.jit.save(modules[key], str(modules_dir / filename))
        log.info(f"Saved {filename}")

    return modules_dir


# ---------------------------------------------------------------------------
# VqVaeKit — convenience wrapper (requires heptokens at load time)
# ---------------------------------------------------------------------------


class VqVaeKit:
    """High-level interface over the four saved TorchScript modules.

    Requires only ``torch`` at load time — no other heptokens dependency.

    Example::

        from heptokens.traced_modules import VqVaeKit
        kit = VqVaeKit.load("output_dir/modules/")

        z_e = kit.encode_float(x, mask)         # input  -> pre-VQ float
        indices = kit.encode_indices(x, mask)   # input  -> int tokens [B,T,Q]
        z_q = kit.lookup(indices)               # tokens -> summed float embedding
        x_hat = kit.decode(z_q, mask)           # float  -> input space
        x_hat = kit.decode_from_indices(indices, mask)  # tokens -> input space
    """

    def __init__(
        self,
        encoder: torch.jit.ScriptModule,
        decoder: torch.jit.ScriptModule,
        codebook_lookup: torch.jit.ScriptModule,
        index_encoder: torch.jit.ScriptModule,
    ) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self._lookup = codebook_lookup
        self._index_encoder = index_encoder

    @classmethod
    def load(cls, modules_dir: str | Path) -> "VqVaeKit":
        """Load all four modules from a ``modules/`` directory.

        Args:
            modules_dir: Path to the directory containing ``encoder.pt``,
                ``decoder.pt``, ``codebook_lookup.pt``, ``index_encoder.pt``.
        """
        d = Path(modules_dir)
        return cls(
            encoder=torch.jit.load(str(d / "encoder.pt")),
            decoder=torch.jit.load(str(d / "decoder.pt")),
            codebook_lookup=torch.jit.load(str(d / "codebook_lookup.pt")),
            index_encoder=torch.jit.load(str(d / "index_encoder.pt")),
        )

    def encode_float(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Map input tokens to pre-quantization float embeddings ``z_e``."""
        return self._encoder(x, mask)

    def encode_indices(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Map input tokens to integer codebook indices (``-1`` = masked)."""
        return self._index_encoder(x, mask)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Sum residual codebook embeddings for the given indices."""
        return self._lookup(indices)

    def decode(self, z_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode float quantized embeddings to input space."""
        return self._decoder(z_q, mask)

    def decode_from_indices(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Decode integer indices all the way to input space."""
        return self.decode(self.lookup(indices), mask)
