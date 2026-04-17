"""Smoke-tests for ZPrimeDataset / ZPrimeMapModule.

Run with:
    python scripts/verify_zprime_dataloader.py

Expects two sample files in data/sample/:
    tops_ZPrime_M1000_W300_batch0.h5
    qcd_Pt300_batch0_partial.h5  (won't be opened; only used for HTTP test)
"""

import sys
from pathlib import Path

import h5py
import numpy as np

# Make the package importable without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from heptokens.data.zprime_mappable import (
    DEFAULT_CST_FEATURES,
    DEFAULT_JET_FEATURES,
    JET_KIN_FEATURES,
    PFCANDS_FEATURES,
    MultiFileDataset,
    ZPrimeDataset,
    ZPrimeMapModule,
)

TOPS_FILE = "data/sample/tops_ZPrime_M1000_W300_batch0.h5"


def check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  OK  : {msg}")


# ── 1. ZPrimeDataset – tops file ──────────────────────────────────────────────
print("\n=== ZPrimeDataset (tops, full) ===")
ds = ZPrimeDataset(TOPS_FILE, label=1)
print(f"  {len(ds)} jets")

sample = ds[0]
check("csts" in sample and "mask" in sample and "jets" in sample and "labels" in sample,
      "sample has expected keys")

n_cst_feats = len(DEFAULT_CST_FEATURES)
check(sample["csts"].shape == (150, n_cst_feats),
      f"csts shape == (150, {n_cst_feats})")
check(sample["mask"].shape == (150,),
      "mask shape == (150,)")
check(sample["mask"].dtype == bool,
      "mask dtype == bool")
check(sample["jets"].shape == (4,),
      "jets shape == (4,)")
check(int(sample["labels"]) == 1,
      "label == 1 (tops)")
check("weights" not in sample,
      "no 'weights' key (tops file has none)")

# Validate mask agrees with the raw "valid" flag
with h5py.File(TOPS_FILE, "r") as f:
    raw_valid = f["PFCands"][0, :, 10].astype(bool)
check(np.array_equal(sample["mask"], raw_valid),
      "mask matches raw valid flag from HDF5")

# Validate jet kinematics
with h5py.File(TOPS_FILE, "r") as f:
    raw_jk = f["jet_kinematics"][0]
check(np.allclose(sample["jets"], raw_jk),
      "jets matches raw jet_kinematics from HDF5")

# Validate constituent features
with h5py.File(TOPS_FILE, "r") as f:
    raw_pf = f["PFCands"][0, :, DEFAULT_CST_FEATURES]
check(np.allclose(sample["csts"], raw_pf),
      "csts matches selected PFCands features from HDF5")

print()

# ── 2. ZPrimeDataset – num_jets / num_csts limits ────────────────────────────
print("=== ZPrimeDataset (num_jets=100, num_csts=40) ===")
ds_small = ZPrimeDataset(TOPS_FILE, label=1, num_jets=100, num_csts=40)
check(len(ds_small) == 100, "len == 100")
check(ds_small[0]["csts"].shape == (40, n_cst_feats), "csts shape == (40, n_feats)")
check(ds_small[0]["mask"].shape == (40,), "mask shape == (40,)")
print()

# ── 3. ZPrimeDataset – custom feature selection ──────────────────────────────
print("=== ZPrimeDataset (kinematics only: px,py,pz,E) ===")
ds_kin = ZPrimeDataset(TOPS_FILE, label=1, cst_feature_indices=[0, 1, 2, 3])
check(ds_kin[0]["csts"].shape == (150, 4), "csts shape == (150, 4)")
print()

# ── 4. MultiFileDataset – two copies of the same file ────────────────────────
print("=== MultiFileDataset (2 x tops file) ===")
mds = MultiFileDataset([TOPS_FILE, TOPS_FILE], label=1)
check(len(mds) == 2 * len(ds), f"len == {2 * len(ds)}")
check(mds.weights is None, "no weights (tops)")
print()

# ── 5. ZPrimeMapModule ────────────────────────────────────────────────────────
print("=== ZPrimeMapModule (tops only, no QCD) ===")
module = ZPrimeMapModule(
    tops_paths=[TOPS_FILE, TOPS_FILE],
    qcd_paths=[TOPS_FILE],        # reuse tops file as fake QCD – just tests plumbing
    num_jets_per_file=200,
    batch_size=32,
    num_workers=0,
)
module.setup("fit")

train_dl = module.train_dataloader()
val_dl   = module.val_dataloader()
test_dl  = module.test_dataloader()

batch = next(iter(train_dl))
check(set(batch.keys()) >= {"csts", "mask", "jets", "labels"},
      "train batch has expected keys")
check(batch["csts"].shape[0] == 32, "train batch size == 32")
check(batch["csts"].shape[2] == n_cst_feats, f"batch cst features == {n_cst_feats}")
check(batch["labels"].dtype == np.int64 or str(batch["labels"].dtype) in ("torch.int64", "int64"),
      "labels dtype int64")
check(module.get_n_classes() == 2, "n_classes == 2")

print()

# ── 6. Weighted QCD loading ───────────────────────────────────────────────────
print("=== ZPrimeDataset (load_weights=True, tops file – no weight key) ===")
ds_w = ZPrimeDataset(TOPS_FILE, label=0, load_weights=True)
check(ds_w.weights is None, "weights is None (no 'weight' dataset in tops file)")
print()

# ── summary ───────────────────────────────────────────────────────────────────
print("All checks passed.\n")
print("Feature reference:")
print("  PFCands features (indices 0-10):")
for i, name in enumerate(PFCANDS_FEATURES):
    marker = "*" if i in DEFAULT_CST_FEATURES else " "
    print(f"    {marker} [{i:2d}] {name}")
print("  jet_kinematics features:")
for i, name in enumerate(JET_KIN_FEATURES):
    print(f"      [{i}] {name}")
