import logging
from functools import partial
from typing import TYPE_CHECKING

import torch as T
import torch.nn as nn
from lightning import LightningModule
from torch.nn.functional import cross_entropy
from torchmetrics import AUROC, Accuracy

from heptokens.data.sequence import LABELS_KEY, MASK_KEY, TOKENS_KEY, TYPE_IDS_KEY
from heptokens.models.sequence_backbone import SequenceBackbone, SequenceBackboneConfig
from heptokens.models.utils import ScheduledOptimiserMixin

if TYPE_CHECKING:
    from heptokens.models.utils import JetBackbone

log = logging.getLogger(__name__)


class Classifier(LightningModule):
    """A class for fine tuning a classifier based on a model with an encoder.

    This should be paired with a scheduler for unfreezing/freezing the backbone.
    """

    def __init__(
        self,
        *,
        data_sample: tuple,
        n_classes: int,
        backbone_path: str,
        class_head: partial,
        optimizer: partial,
        scheduler: partial,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_classes = n_classes

        # Load the pretrained and pickled JetBackbone object.
        log.info(f"Loading backbone from {backbone_path}")
        self.backbone: JetBackbone = T.load(backbone_path, map_location="cpu")

        # Create the head for the downstream task
        self.class_head = class_head(
            inpt_dim=self.backbone.encoder.outp_dim,
            outp_dim=n_classes,
        )

        # Metrics
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)

    def forward(
        self,
        csts: T.Tensor,
        mask: T.BoolTensor,
        jets: T.Tensor,
    ) -> T.Tensor:
        x, mask = self.backbone(csts, mask, jets)  # Might gain registers
        return self.class_head(x, mask=mask)

    def _shared_step(self, data: tuple, prefix: str) -> T.Tensor:
        """Shared step used in both training and validaiton."""
        csts = data["csts"]
        labels = data["labels"]
        mask = data["mask"]
        jets = data["jets"]

        output = self.forward(csts, mask, jets)
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

    def predict_step(self, data: dict) -> T.Tensor:
        """Return a dictionary for saving final exported scores."""
        csts = data["csts"]
        labels = data["labels"]
        mask = data["mask"]
        jets = data["jets"]
        output = self.forward(csts, mask, jets)
        return {"output": output, "label": labels.unsqueeze(-1)}

    def configure_optimizers(self) -> dict:
        params = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.hparams.optimizer(params)
        sched = self.hparams.scheduler(optimizer=opt, model=self)
        return [opt], [{"scheduler": sched, "interval": "step"}]


class SequenceClassifier(nn.Module):
    """Classification head on top of a token-sequence backbone."""

    def __init__(
        self,
        backbone: "SequenceBackbone",
        num_classes: int = 2,
        freeze_backbone: bool = False,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.pooling = pooling
        hidden_dim = backbone.config.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        tokens: T.Tensor,
        mask: T.Tensor,
        type_ids: T.Tensor | None = None,
    ) -> T.Tensor:
        hidden = self.backbone(tokens, mask, type_ids)
        pooled = self._pool(hidden, mask)
        return self.classifier(pooled)

    def _pool(self, hidden: T.Tensor, mask: T.Tensor) -> T.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "mean":
            valid = mask.bool().unsqueeze(-1)
            return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        if self.pooling == "max":
            valid = mask.bool().unsqueeze(-1)
            return hidden.masked_fill(~valid, float("-inf")).max(dim=1).values
        raise ValueError(f"Unknown sequence pooling mode: {self.pooling}")

    def forward_batch(self, batch: dict) -> T.Tensor:
        return self(
            batch[TOKENS_KEY],
            batch[MASK_KEY],
            batch.get(TYPE_IDS_KEY),
        )


class LitSequenceClassifier(ScheduledOptimiserMixin, LightningModule):
    """Lightning classifier for token-sequence batches."""

    def __init__(
        self,
        *,
        data_sample: dict | None = None,
        n_classes: int,
        backbone_ckpt_path: str | None = None,
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
        use_type_embedding: bool = True,
        use_position_embedding: bool = True,
        pooling: str = "cls",
        freeze_backbone: bool = False,
        learning_rate: float = 1e-4,
        optimizer=None,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["data_sample"])
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        config = SequenceBackboneConfig(
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
            use_type_embedding=use_type_embedding,
            use_position_embedding=use_position_embedding,
        )
        backbone = SequenceBackbone(config)
        if backbone_ckpt_path:
            self._load_backbone(backbone, backbone_ckpt_path)
        self.model = SequenceClassifier(
            backbone,
            num_classes=n_classes,
            freeze_backbone=freeze_backbone,
            pooling=pooling,
        )
        self.train_acc = Accuracy("multiclass", num_classes=n_classes)
        self.valid_acc = Accuracy("multiclass", num_classes=n_classes)
        self.train_auc = AUROC(task="multiclass", num_classes=n_classes, average="macro")
        self.valid_auc = AUROC(task="multiclass", num_classes=n_classes, average="macro")

    def _load_backbone(self, backbone: SequenceBackbone, ckpt_path: str) -> None:
        checkpoint = T.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefixes = (
            "model.backbone.",
            "backbone.",
            "model.",
        )
        backbone_state = {}
        for key, value in state_dict.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    stripped = key.removeprefix(prefix)
                    if stripped in backbone.state_dict():
                        backbone_state[stripped] = value
                    break
        if not backbone_state:
            backbone_state = state_dict
        backbone.load_state_dict(backbone_state, strict=False)

    def forward(self, batch: dict) -> T.Tensor:
        return self.model.forward_batch(batch)

    def _shared_step(self, batch: dict, prefix: str) -> T.Tensor:
        labels = batch[LABELS_KEY]
        output = self.forward(batch)
        loss = cross_entropy(output, labels, label_smoothing=0.1)
        self.log(f"{prefix}/total_loss", loss, prog_bar=True)

        acc = getattr(self, f"{prefix}_acc")
        acc(output, labels)
        self.log(f"{prefix}/acc", acc, prog_bar=True)

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
        labels = batch[LABELS_KEY]
        return {"output": output, "label": labels.unsqueeze(-1)}
