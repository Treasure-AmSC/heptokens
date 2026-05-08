"""Datasets and dataloader helpers for token sequence parquet files."""
# TODO: Replace this Parquet-only adapter with an event-level tokenization datamodule
# once object tokenizers are integrated into the training pipeline.

from __future__ import annotations

import logging
import platform
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from heptokens.data.sequence import (
    LABELS_KEY,
    MASK_KEY,
    TOKENS_KEY,
    TYPE_IDS_KEY,
    first_existing_column,
)
from heptokens.data.atlas_mappable import BaseMapModule

log = logging.getLogger(__name__)
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2


class TokenParquetDataset(Dataset):
    """Load tokenized sequences from parquet files.

    The dataset normalizes parquet column names to a generic sequence batch:
    ``tokens``, ``mask``, optional ``type_ids``, and optional ``labels``.
    Preferred parquet columns are ``tokens``, ``mask``, and optional ``type_ids``.
    Legacy columns from the old standalone tokenizer are also accepted:
    ``input_ids``, ``attention_mask``, and ``token_type_ids``.
    """

    def __init__(
        self,
        parquet_path: str,
        label: int | None = None,
        *,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        label_column: str | None = None,
    ) -> None:
        import pyarrow.parquet as pq

        log.info("Loading %s%s...", parquet_path, "" if label is None else f" (label={label})")
        t0 = time.time()
        table = pq.read_table(parquet_path)
        self.n = len(table)
        token_column = token_column or first_existing_column(
            table.column_names,
            ["tokens", "input_ids"],
        )
        mask_column = mask_column or first_existing_column(
            table.column_names,
            ["mask", "attention_mask"],
        )
        type_column = type_column or first_existing_column(
            table.column_names,
            ["type_ids", "token_type_ids"],
            required=False,
        )

        self.tokens = np.array(table[token_column].to_pylist(), dtype=np.int64)
        self.mask = np.array(table[mask_column].to_pylist(), dtype=bool)

        if type_column and type_column in table.column_names:
            self.type_ids = np.array(table[type_column].to_pylist(), dtype=np.int64)
        else:
            self.type_ids = np.zeros_like(self.tokens, dtype=np.int64)

        if label is not None:
            self.labels = np.full(self.n, label, dtype=np.int64)
        elif label_column and label_column in table.column_names:
            self.labels = np.array(table[label_column].to_pylist(), dtype=np.int64)
        else:
            self.labels = None

        log.info("  Loaded %s sequences in %.1fs", self.n, time.time() - t0)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        batch = {
            TOKENS_KEY: torch.tensor(self.tokens[idx], dtype=torch.long),
            MASK_KEY: torch.tensor(self.mask[idx], dtype=torch.bool),
            TYPE_IDS_KEY: torch.tensor(self.type_ids[idx], dtype=torch.long),
        }
        if self.labels is not None:
            batch[LABELS_KEY] = torch.tensor(self.labels[idx], dtype=torch.long)
        return batch


def make_classification_loaders(
    signal_parquets: list[str],
    background_parquets: list[str],
    *,
    batch_size: int = 256,
    max_sequences: int | None = None,
    seed: int = 42,
    token_column: str | None = None,
    mask_column: str | None = None,
    type_column: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_kwargs = {
        "token_column": token_column,
        "mask_column": mask_column,
        "type_column": type_column,
    }
    datasets = [TokenParquetDataset(fp, label=1, **dataset_kwargs) for fp in signal_parquets]
    datasets.extend(
        TokenParquetDataset(fp, label=0, **dataset_kwargs) for fp in background_parquets
    )
    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if max_sequences and len(full) > max_sequences:
        full, _ = random_split(
            full,
            [max_sequences, len(full) - max_sequences],
            generator=torch.Generator().manual_seed(99),
        )

    total = len(full)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_set, val_set, test_set = random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    kwargs = {"batch_size": batch_size, "num_workers": NUM_WORKERS}
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )


def make_pretrain_loaders(
    parquet_files: list[str],
    *,
    batch_size: int = 256,
    max_sequences: int | None = None,
    seed: int = 42,
    token_column: str | None = None,
    mask_column: str | None = None,
    type_column: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    dataset_kwargs = {
        "token_column": token_column,
        "mask_column": mask_column,
        "type_column": type_column,
    }
    datasets = [TokenParquetDataset(fp, **dataset_kwargs) for fp in parquet_files]
    full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if max_sequences and len(full) > max_sequences:
        full, _ = random_split(
            full,
            [max_sequences, len(full) - max_sequences],
            generator=torch.Generator().manual_seed(99),
        )

    total = len(full)
    train_size = int(0.9 * total)
    val_size = total - train_size
    train_set, val_set = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    kwargs = {"batch_size": batch_size, "num_workers": NUM_WORKERS}
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
    )


class TokenParquetClassificationModule(BaseMapModule):
    """DataModule for labelled signal/background token parquet files."""

    def __init__(
        self,
        *,
        signal_parquets: list[str],
        background_parquets: list[str],
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
        max_sequences: int | None = None,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

        dataset_kwargs = {
            "token_column": token_column,
            "mask_column": mask_column,
            "type_column": type_column,
        }
        datasets = [TokenParquetDataset(fp, label=1, **dataset_kwargs) for fp in signal_parquets]
        datasets.extend(
            TokenParquetDataset(fp, label=0, **dataset_kwargs) for fp in background_parquets
        )
        full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        if max_sequences and len(full) > max_sequences:
            full, _ = random_split(
                full,
                [max_sequences, len(full) - max_sequences],
                generator=torch.Generator().manual_seed(seed),
            )

        total = len(full)
        train_size = int(train_frac * total)
        val_size = int(val_frac * total)
        test_size = total - train_size - val_size
        self.train_set, self.valid_set, self.test_set = random_split(
            full,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed),
        )

    def setup(self, stage: str) -> None:
        pass


class TokenParquetPretrainModule(BaseMapModule):
    """DataModule for unlabeled masked-sequence pretraining parquet files."""

    def __init__(
        self,
        *,
        parquet_files: list[str],
        train_frac: float = 0.9,
        val_frac: float = 0.1,
        seed: int = 42,
        max_sequences: int | None = None,
        token_column: str | None = None,
        mask_column: str | None = None,
        type_column: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not abs(train_frac + val_frac - 1.0) < 1e-6:
            raise ValueError("train_frac + val_frac must sum to 1.0")

        dataset_kwargs = {
            "token_column": token_column,
            "mask_column": mask_column,
            "type_column": type_column,
        }
        datasets = [TokenParquetDataset(fp, **dataset_kwargs) for fp in parquet_files]
        full = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        if max_sequences and len(full) > max_sequences:
            full, _ = random_split(
                full,
                [max_sequences, len(full) - max_sequences],
                generator=torch.Generator().manual_seed(seed),
            )

        total = len(full)
        train_size = int(train_frac * total)
        val_size = total - train_size
        self.train_set, self.valid_set = random_split(
            full,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        self.test_set = self.valid_set

    def setup(self, stage: str) -> None:
        pass
