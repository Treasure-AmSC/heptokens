import logging
from typing import Literal

import torch as T
from lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy

from gdig.data.preprocessing import VqvaeTokenizer
from gdig.models.transformer import Transformer

log = logging.getLogger(__name__)


class TokenClassifier(LightningModule):
    """Classifier that operates on VQ-VAE token IDs."""

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
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.num_quantizers = num_quantizers
        self.pooling = pooling

        # Add tokenizer if checkpoint provided
        if tokenizer_ckpt is not None:
            self.tokenizer = VqvaeTokenizer(tokenizer_ckpt)
        else:
            self.tokenizer = None

        # +1 for padding index 0 (tokens are shifted by +1)
        self.token_emb = T.nn.Embedding(codebook_size + 1, d_model, padding_idx=0)

        self.encoder = Transformer(
            input_dim=d_model,
            output_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        if pooling == "cls":
            self.cls_token = T.nn.Parameter(T.zeros(1, 1, d_model))

        self.classifier = T.nn.Linear(d_model, n_classes)

        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def _embed_tokens(self, tokens: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        """Embed token ids and combine quantizers if present."""
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(-1)

        # Shift so padding is 0; set masked positions to 0
        tokens = tokens + 1
        tokens = T.where(mask.unsqueeze(-1), tokens, T.zeros_like(tokens))

        # Embed each quantizer and sum across quantizer dimension
        emb = self.token_emb(tokens)  # [B, N, Q, D]
        emb = emb.sum(dim=2)

        return emb

    def forward(self, tokens: T.Tensor, mask: T.BoolTensor) -> T.Tensor:
        if self.pooling == "cls":
            x = self._embed_tokens(tokens, mask)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = T.cat([cls, x], dim=1)
            cls_mask = T.ones((mask.shape[0], 1), device=mask.device, dtype=mask.dtype)
            mask = T.cat([cls_mask, mask], dim=1)
            x = self.encoder(x, mask=mask)
            pooled = x[:, 0, :]
        else:
            x = self._embed_tokens(tokens, mask)
            x = self.encoder(x, mask=mask)
            if self.pooling == "mean":
                valid_sum = (x * mask.unsqueeze(-1)).sum(dim=1)
                valid_count = mask.sum(dim=1, keepdim=True)
                pooled = valid_sum / valid_count.clamp(min=1)
            elif self.pooling == "max":
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                pooled = x_masked.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")

        return self.classifier(pooled)

    def process_data(self, batch: dict) -> dict:
        """Tokenize input data if tokenizer is available."""
        if self.tokenizer is not None:
            batch = self.tokenizer(batch)
        return batch

    def _shared_step(self, data: dict, prefix: str) -> T.Tensor:
        data = self.process_data(data)
        tokens = data["tokens"]
        labels = data["labels"]
        mask = data["mask"]
        output = self.forward(tokens, mask)
        loss = cross_entropy(output, labels, label_smoothing=0.1)
        self.log(f"{prefix}/total_loss", loss)

        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)
        self.log(f"{prefix}/acc", acc)
        return loss

    def training_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "train")

    def validation_step(self, data: dict) -> T.Tensor:
        return self._shared_step(data, "valid")

    def predict_step(self, data: dict) -> dict:
        data = self.process_data(data)
        tokens = data["tokens"]
        labels = data["labels"]
        mask = data["mask"]
        output = self.forward(tokens, mask)
        return {"output": output, "label": labels.unsqueeze(-1)}

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
