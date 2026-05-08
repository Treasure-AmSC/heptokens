"""Masked sequence modelling head for foundation-model pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from heptokens.data.sequence import MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.sequence_backbone import SequenceBackbone, SequenceBackboneConfig
from heptokens.models.utils import ScheduledOptimiserMixin


class MaskedSequenceModel(nn.Module):
    """Masked sequence modelling head, analogous to MLM for tokenized sequences."""

    def __init__(self, config: SequenceBackboneConfig):
        super().__init__()
        self.config = config
        self.backbone = SequenceBackbone(config)
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.vocab_size),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.backbone(tokens, mask, type_ids)
        logits = self.mlm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @staticmethod
    def mask_tokens(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        mask_prob: float = 0.15,
        mask_token_id: int = 3,
        vocab_size: int = 20000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = tokens.clone()
        labels[:] = -100
        special = tokens <= 3
        can_mask = mask.bool() & ~special

        prob_matrix = torch.full_like(tokens, mask_prob, dtype=torch.float)
        prob_matrix[~can_mask] = 0.0
        masked_indices = torch.bernoulli(prob_matrix).bool()
        labels[masked_indices] = tokens[masked_indices]

        replace_mask = torch.bernoulli(torch.full_like(prob_matrix, 0.8)).bool() & masked_indices
        tokens[replace_mask] = mask_token_id

        random_mask = (
            torch.bernoulli(torch.full_like(prob_matrix, 0.5)).bool()
            & masked_indices
            & ~replace_mask
        )
        random_tokens = torch.randint(4, vocab_size, tokens.shape, device=tokens.device)
        tokens[random_mask] = random_tokens[random_mask]
        return tokens, labels

    def forward_batch(
        self,
        batch: dict,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self(
            batch[TOKENS_KEY],
            batch[MASK_KEY],
            batch.get(TYPE_IDS_KEY),
            labels=labels,
        )


# Backwards-compatible names for older imports.
MaskedEventModel = MaskedSequenceModel


class LitMaskedSequenceModel(ScheduledOptimiserMixin, LightningModule):
    """Lightning wrapper for masked sequence modelling pretraining."""

    def __init__(
        self,
        *,
        data_sample: dict | None = None,
        n_classes: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 128,
        vocab_size: int = 20000,
        num_type_ids: int = 12,
        mask_token_id: int = 3,
        pad_token_id: int = 0,
        mask_prob: float = 0.15,
        hierarchical: bool = False,
        n_groups: int = 3,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.config = SequenceBackboneConfig(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
            vocab_size=vocab_size,
            num_type_ids=num_type_ids,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            mask_prob=mask_prob,
            hierarchical=hierarchical,
            n_groups=n_groups,
        )
        self.model = MaskedSequenceModel(self.config)

    @property
    def backbone(self) -> SequenceBackbone:
        return self.model.backbone

    def forward(self, tokens, mask, type_ids=None, labels=None):
        return self.model(tokens, mask, type_ids, labels=labels)

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        tokens = batch[TOKENS_KEY].clone()
        mask = batch[MASK_KEY]
        type_ids = batch.get(TYPE_IDS_KEY)
        masked_tokens, labels = MaskedSequenceModel.mask_tokens(
            tokens,
            mask,
            mask_prob=self.config.mask_prob,
            mask_token_id=self.config.mask_token_id,
            vocab_size=self.config.vocab_size,
        )
        logits, loss = self(masked_tokens, mask, type_ids, labels=labels)
        mask_pos = labels != -100
        if mask_pos.any():
            acc = (logits.argmax(dim=-1)[mask_pos] == labels[mask_pos]).float().mean()
            self.log(f"{prefix}/mask_acc", acc, prog_bar=True)
        self.log(f"{prefix}/total_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict) -> torch.Tensor:
        return self._shared_step(batch, "valid")
