# TODO: at some point you need to integrate this with classifier.py which support fine tuning.
import logging
from abc import ABC, abstractmethod
from typing import Literal

import torch as T
from lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import AUROC, Accuracy

from heptokens.data.collation import VqvaeTokenizer
from heptokens.models.transformer import Transformer
from heptokens.models.utils import ScheduledOptimiserMixin

log = logging.getLogger(__name__)


class Embedder(ABC, T.nn.Module):
    """Base class for sequence embedders.

    Embedders convert raw input data into sequence embeddings.
    They handle both data preprocessing and embedding logic.
    """

    @abstractmethod
    def embed(self, batch: dict) -> tuple[T.Tensor, T.BoolTensor]:
        """Convert batch to embeddings.

        Args:
            batch: Input batch dictionary (keys depend on embedder type)

        Returns:
            (embeddings, mask) where:
            - embeddings: [B, N, d_model] tensor
            - mask: [B, N] boolean tensor (True = valid)
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Embedding output dimension."""
        pass


class SequenceEncoder(ABC, T.nn.Module):
    """Base class for sequence encoding (e.g., attention-based)."""

    @abstractmethod
    def encode(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        """Encode sequence.

        Args:
            x: [B, N, input_dim]
            mask: [B, N] boolean mask

        Returns:
            [B, N, output_dim]
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension of encoder."""
        pass


class Pooler(ABC, T.nn.Module):
    """Base class for sequence pooling strategies."""

    @abstractmethod
    def pool(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        """Pool sequence representation.

        Args:
            x: [B, N, d_model]
            mask: [B, N] boolean mask

        Returns:
            [B, d_model]
        """
        pass


class TransformerEncoder(SequenceEncoder):
    """Transformer-based sequence encoder."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.transformer = Transformer(
            input_dim=input_dim,
            output_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    @property
    def output_dim(self) -> int:
        return self.d_model

    def encode(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        return self.transformer(x, mask=mask)


class ClsTokenPooler(Pooler):
    """Pooling using learnable CLS token."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.cls_token = T.nn.Parameter(T.zeros(1, 1, d_model))

    def pool(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        # Prepend CLS token
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = T.cat([cls, x], dim=1)
        cls_mask = T.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
        mask = T.cat([cls_mask, mask], dim=1)
        # CLS representation is at position 0
        return x[:, 0, :]


class MeanPooler(Pooler):
    """Mean pooling over valid positions."""

    def pool(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        valid_sum = (x * mask.unsqueeze(-1)).sum(dim=1)
        valid_count = mask.sum(dim=1, keepdim=True)
        return valid_sum / valid_count.clamp(min=1)


class MaxPooler(Pooler):
    """Max pooling over valid positions."""

    def pool(self, x: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return x_masked.max(dim=1)[0]


class TokenEmbedder(Embedder):
    """Embedder for VQ-VAE token IDs with configurable aggregation."""

    def __init__(
        self,
        codebook_size: int,
        d_model: int,
        num_quantizers: int = 1,
        tokenizer_ckpt: str | None = None,
        aggregate: Literal["sum", "flatten", "learned_weights", "attention"] = "sum",
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.num_quantizers = num_quantizers
        self.tokenizer_ckpt = tokenizer_ckpt
        self.aggregate = aggregate

        self.tokenizer = None
        if tokenizer_ckpt is not None:
            self.tokenizer = VqvaeTokenizer(tokenizer_ckpt)

        # Compute embedding dimension per quantizer
        if aggregate == "flatten":
            import math

            emb_dim_per_q = math.ceil(d_model / num_quantizers)
            self.token_emb = T.nn.Embedding(codebook_size + 1, emb_dim_per_q, padding_idx=0)
            self._output_dim = d_model
        else:
            self.token_emb = T.nn.Embedding(codebook_size + 1, d_model, padding_idx=0)
            self._output_dim = d_model

        # Aggregation-specific layers
        if aggregate == "learned_weights":
            # Learnable weights for each quantizer: [num_quantizers]
            self.quantizer_weights = T.nn.Parameter(T.ones(num_quantizers) / num_quantizers)
        elif aggregate == "attention":
            # Multi-head attention to combine quantizers
            self.quantizer_attention = T.nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.0,
            )
            # Learnable importance weights for each quantizer
            self.quantizer_weights = T.nn.Parameter(T.ones(num_quantizers) / num_quantizers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def embed(self, batch: dict) -> tuple[T.Tensor, T.BoolTensor]:
        # Preprocess if tokenizer available
        if self.tokenizer is not None:
            batch = self.tokenizer(batch)

        tokens = batch["tokens"]
        mask = batch["mask"]

        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(-1)

        # Shift for padding index and mask out invalid positions
        tokens = tokens + 1
        tokens = T.where(mask.unsqueeze(-1), tokens, T.zeros_like(tokens))

        # Embed each quantizer: [B, N, Q, D]
        emb = self.token_emb(tokens)

        if self.aggregate == "flatten":
            # Flatten quantizers: [B, N, Q, D_per_Q] -> [B, N, Q*D_per_Q]
            emb = emb.reshape(emb.shape[0], emb.shape[1], -1)
            # Slice to actual output dimension
            emb = emb[:, :, : self._output_dim]

        elif self.aggregate == "learned_weights":
            # Apply learned weights: [B, N, Q, D] * [Q] -> [B, N, D]
            weights = self.quantizer_weights.view(1, 1, -1, 1)
            emb = (emb * weights).sum(dim=2)

        elif self.aggregate == "attention":
            batch_size, seq_len, num_q, d_model = emb.shape
            emb_reshaped = emb.reshape(batch_size * seq_len, num_q, d_model)

            # Apply learned per-quantizer scaling
            weights = T.softmax(self.quantizer_weights, dim=0)  # [Q]
            emb_reshaped = emb_reshaped * weights.view(1, num_q, 1)  # Scale each quantizer

            # Expand mask to quantizer dimension: [B, N] -> [B, N, Q] -> [B*N, Q]
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_q)  # [B, N, Q]
            mask_reshaped = mask_expanded.reshape(batch_size * seq_len, num_q)  # [B*N, Q]
            emb_reshaped = emb_reshaped * mask_reshaped.unsqueeze(-1).float()

            # Apply attention
            attn_out, _ = self.quantizer_attention(
                emb_reshaped,
                emb_reshaped,
                emb_reshaped,
                need_weights=False,
            )

            # Mean pool over quantizers (only valid ones)
            valid_count = mask_reshaped.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B*N, 1]
            emb = (attn_out * mask_reshaped.float().unsqueeze(-1)).sum(dim=1) / valid_count

            emb = emb.reshape(batch_size, seq_len, d_model)

        else:  # sum (default)
            # Sum over quantizers: [B, N, Q, D] -> [B, N, D]
            emb = emb.sum(dim=2)

        return emb, mask


class FeatureEmbedder(Embedder):
    """Embedder for raw constituent features."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.projection = T.nn.Linear(input_dim, d_model)

    @property
    def output_dim(self) -> int:
        return self.d_model

    def embed(self, batch: dict) -> tuple[T.Tensor, T.BoolTensor]:
        csts = batch["csts"]
        mask = batch["mask"]

        # Zero out masked positions
        csts = T.where(mask.unsqueeze(-1), csts, T.zeros_like(csts))
        embeddings = self.projection(csts)

        return embeddings, mask


class VectorEmbedder(Embedder):
    """Embedder using VQ-VAE codebook vectors (z_q) instead of token IDs."""

    def __init__(
        self,
        codebook_dim: int,
        d_model: int,
        tokenizer_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.codebook_dim = codebook_dim
        self.d_model = d_model
        self.projection = T.nn.Linear(codebook_dim, d_model)

        self.tokenizer = None
        if tokenizer_ckpt is not None:
            self.tokenizer = VqvaeTokenizer(tokenizer_ckpt)

    @property
    def output_dim(self) -> int:
        return self.d_model

    def embed(self, batch: dict) -> tuple[T.Tensor, T.BoolTensor]:
        if self.tokenizer is not None:
            batch = self.tokenizer(batch)

        z_q = batch["z_q"]       # [B, N, codebook_dim]
        mask = batch["mask"]     # [B, N]

        z_q = T.where(mask.unsqueeze(-1), z_q, T.zeros_like(z_q))
        embeddings = self.projection(z_q)

        return embeddings, mask


class JetClassifier(ScheduledOptimiserMixin, LightningModule):
    """General-purpose jet classifier with pluggable components.

    Architecture:
        Input -> Embedder -> Encoder -> Pooler -> Head -> Logits
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        encoder: SequenceEncoder,
        pooler: Pooler,
        n_classes: int,
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "encoder", "pooler"])

        # Validate compatibility
        if embedder.output_dim != encoder.encode.__code__.co_varnames[1:2][0]:  # Quick check
            log.warning(
                f"Embedder output_dim ({embedder.output_dim}) may not match "
                f"encoder input_dim. Check compatibility."
            )

        self.embedder = embedder
        self.encoder = encoder
        self.pooler = pooler
        self.n_classes = n_classes

        # Classification head
        self.classifier = T.nn.Linear(encoder.output_dim, n_classes)

        # Metrics
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

        # AUC metrics (one-vs-rest for multiclass)
        self.train_auc = AUROC(task="multiclass", num_classes=n_classes, average="macro")
        self.valid_auc = AUROC(task="multiclass", num_classes=n_classes, average="macro")

        # Store outputs for ROC plotting
        self.validation_outputs = []

    def forward(self, batch: dict) -> T.Tensor:
        """Forward pass through entire pipeline."""
        # Embed
        x, mask = self.embedder.embed(batch)

        # Encode
        x = self.encoder.encode(x, mask)

        # Pool
        x = self.pooler.pool(x, mask)

        # Classify
        return self.classifier(x)

    def _shared_step(self, batch: dict, prefix: str) -> T.Tensor:
        labels = batch["labels"]

        output = self.forward(batch)
        loss = cross_entropy(output, labels, label_smoothing=0.1)
        self.log(f"{prefix}/total_loss", loss)

        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)
        self.log(f"{prefix}/acc", acc)

        # Track AUC
        auc = getattr(self, f"{prefix}_auc")
        probs = T.softmax(output, dim=1)
        auc(probs, labels)
        self.log(f"{prefix}/auc", auc)

        return loss

    def training_step(self, batch: dict) -> T.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> T.Tensor:
        return self._shared_step(batch, "valid")

    def predict_step(self, batch: dict) -> dict:
        output = self.forward(batch)
        labels = batch["labels"]
        return {"output": output, "label": labels.unsqueeze(-1)}


class TokenClassifier(JetClassifier):
    """Explicit instantiation of JetClassifier using VQ-VAE token embedder."""

    def __init__(
        self,
        *,
        data_sample: tuple | None = None,
        n_classes: int,
        codebook_size: int,
        num_quantizers: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: Literal["mean", "max", "cls"] = "mean",
        learning_rate: float = 1e-3,
        tokenizer_ckpt: str | None = None,
        aggregate_tokens: Literal["sum", "flatten", "learned_weights", "attention"] = "flatten",
        quantizer_attention_heads: int = 4,
        **kwargs,
    ) -> None:
        embedder = TokenEmbedder(
            codebook_size=codebook_size,
            d_model=d_model,
            num_quantizers=num_quantizers,
            tokenizer_ckpt=tokenizer_ckpt,
            aggregate=aggregate_tokens,
            num_heads=quantizer_attention_heads,
        )
        encoder = TransformerEncoder(
            input_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        pooler_cls = {
            "cls": ClsTokenPooler,
            "mean": MeanPooler,
            "max": MaxPooler,
        }[pooling]
        pooler = pooler_cls(d_model) if pooling == "cls" else pooler_cls()

        super().__init__(
            embedder=embedder,
            encoder=encoder,
            pooler=pooler,
            n_classes=n_classes,
            learning_rate=learning_rate,
            **kwargs,
        )


class FeatureClassifier(JetClassifier):
    """Explicit instantiation of JetClassifier using raw feature embedder."""

    def __init__(
        self,
        *,
        data_sample: tuple | None = None,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: Literal["mean", "max", "cls"] = "mean",
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> None:
        embedder = FeatureEmbedder(
            input_dim=data_sample["csts"].shape[-1],
            d_model=d_model,
        )
        encoder = TransformerEncoder(
            input_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        pooler_cls = {
            "cls": ClsTokenPooler,
            "mean": MeanPooler,
            "max": MaxPooler,
        }[pooling]
        pooler = pooler_cls(d_model) if pooling == "cls" else pooler_cls()

        super().__init__(
            embedder=embedder,
            encoder=encoder,
            pooler=pooler,
            n_classes=n_classes,
            learning_rate=learning_rate,
            **kwargs,
        )


class VectorClassifier(JetClassifier):
    """Classifier using VQ-VAE codebook vectors (z_q) as input."""

    def __init__(
        self,
        *,
        data_sample: tuple | None = None,
        n_classes: int,
        codebook_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        activation: str = "gelu",
        pooling: Literal["mean", "max", "cls"] = "mean",
        learning_rate: float = 1e-3,
        tokenizer_ckpt: str | None = None,
        **kwargs,
    ) -> None:
        embedder = VectorEmbedder(
            codebook_dim=codebook_dim,
            d_model=d_model,
            tokenizer_ckpt=tokenizer_ckpt,
        )
        encoder = TransformerEncoder(
            input_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        pooler_cls = {
            "cls": ClsTokenPooler,
            "mean": MeanPooler,
            "max": MaxPooler,
        }[pooling]
        pooler = pooler_cls(d_model) if pooling == "cls" else pooler_cls()

        super().__init__(
            embedder=embedder,
            encoder=encoder,
            pooler=pooler,
            n_classes=n_classes,
            learning_rate=learning_rate,
            **kwargs,
        )
