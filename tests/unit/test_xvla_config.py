import argparse
import json

import pytest


def test_parse_bool_variants(xvla_server_module):
    assert xvla_server_module.parse_bool("true") is True
    assert xvla_server_module.parse_bool("1") is True
    assert xvla_server_module.parse_bool("YES") is True
    assert xvla_server_module.parse_bool("false") is False
    assert xvla_server_module.parse_bool("0") is False


def test_parse_bool_rejects_invalid(xvla_server_module):
    with pytest.raises(argparse.ArgumentTypeError):
        xvla_server_module.parse_bool("maybe")


def test_load_user_config_supports_flat_or_nested_model_block(xvla_server_module, tmp_path):
    flat_path = tmp_path / "flat.json"
    flat_path.write_text(json.dumps({"num_images": 2, "use_proprio": False}), encoding="utf-8")
    assert xvla_server_module._load_user_config(str(flat_path)) == {"num_images": 2, "use_proprio": False}

    nested_path = tmp_path / "nested.json"
    nested_path.write_text(json.dumps({"model": {"proprio_dim": 12, "unnorm_key": "demo"}}), encoding="utf-8")
    assert xvla_server_module._load_user_config(str(nested_path)) == {
        "proprio_dim": 12,
        "unnorm_key": "demo",
    }


def test_load_user_config_missing_file_is_optional(xvla_server_module, tmp_path):
    assert xvla_server_module._load_user_config(str(tmp_path / "missing.json")) == {}


def test_apply_user_overrides_json_takes_precedence(xvla_server_module, namespace_defaults, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "num_images": 4,
                "use_proprio": "false",
                "proprio_dim": 7,
                "unnorm_key": "custom",
            }
        ),
        encoding="utf-8",
    )

    args = namespace_defaults
    args.num_images = 1
    args.use_proprio = True
    args.proprio_dim = 10
    args.unnorm_key = ""
    args.user_config = str(config_path)

    updated = xvla_server_module._apply_user_overrides(args)
    assert updated.num_images == 4
    assert updated.use_proprio is False
    assert updated.proprio_dim == 7
    assert updated.unnorm_key == "custom"


def test_apply_user_overrides_invalid_values_fail_fast(xvla_server_module, namespace_defaults, tmp_path):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({"num_images": 0}), encoding="utf-8")

    args = namespace_defaults
    args.user_config = str(config_path)

    with pytest.raises(RuntimeError, match="num_images"):
        xvla_server_module._apply_user_overrides(args)
