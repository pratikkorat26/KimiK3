"""tests/test_config_yaml.py — YAML load/save roundtrip for ModelConfig.

Contract:
  1. to_yaml → from_yaml reproduces the config exactly (all presets)
  2. the shipped configs/*.yaml files load and match their presets
  3. unknown keys in a YAML file are ignored (forward-compat)
"""

from pathlib import Path

import pytest

from kimi_k3 import ModelConfig

PRESETS = {
    "tiny": ModelConfig.tiny,
    "tiny_hybrid": ModelConfig.tiny_hybrid,
    "small": ModelConfig.small,
    "kimi_1b_64k": ModelConfig.kimi_1b_64k,
}
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_yaml_roundtrip(name, tmp_path):
    cfg = PRESETS[name]()
    path = tmp_path / f"{name}.yaml"
    cfg.to_yaml(str(path))
    loaded = ModelConfig.from_yaml(str(path))
    assert loaded.to_dict() == cfg.to_dict()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_shipped_configs_match_presets(name):
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{path.name} not shipped")
    loaded = ModelConfig.from_yaml(str(path))
    assert loaded.to_dict() == PRESETS[name]().to_dict()


def test_from_dict_ignores_unknown_keys():
    d = ModelConfig.tiny().to_dict()
    d["some_future_field"] = 123
    cfg = ModelConfig.from_dict(d)
    assert not hasattr(cfg, "some_future_field")
