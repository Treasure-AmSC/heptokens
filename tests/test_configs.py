"""Verify bundled Hydra configs point to importable targets."""

import importlib
from pathlib import Path

import pytest
import yaml

import heptokens

CONF_DIR = Path(heptokens.__file__).parent / "conf"

TEMPLATE_FILES = {"new_modality_encode.yaml", "new_modality_vqvae.yaml"}


def _collect_targets(obj, targets=None):
    """Recursively collect all _target_ values from a parsed YAML structure."""
    if targets is None:
        targets = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "_target_":
                targets.append(val)
            else:
                _collect_targets(val, targets)
    elif isinstance(obj, list):
        for item in obj:
            _collect_targets(item, targets)
    return targets


def _gather_config_targets():
    """Yield (config_path, target_string) pairs for all non-template bundled configs."""
    for yaml_path in CONF_DIR.rglob("*.yaml"):
        if yaml_path.name in TEMPLATE_FILES:
            continue
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if data is None:
            continue
        for target in _collect_targets(data):
            if "#" in target:
                continue
            yield yaml_path.relative_to(CONF_DIR), target


_ALL_TARGETS = list(_gather_config_targets())


@pytest.mark.parametrize(
    "config_path,target",
    _ALL_TARGETS,
    ids=[f"{p}::{t.rsplit('.', 1)[-1]}" for p, t in _ALL_TARGETS],
)
def test_target_is_importable(config_path, target):
    """Each _target_ in bundled configs must resolve to an importable Python object."""
    module_path, attr_name = target.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr_name), (
        f"{config_path}: _target_ '{target}' — module '{module_path}' "
        f"has no attribute '{attr_name}'"
    )
