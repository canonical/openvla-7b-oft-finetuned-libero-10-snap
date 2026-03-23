import json
from types import SimpleNamespace

import numpy as np
import pytest


def _response_json(response):
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "body"):
        return json.loads(response.body.decode("utf-8"))
    raise AssertionError("Unsupported response type")


def test_deserialize_image_payload_from_3d_array(xvla_server_module):
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    parsed = xvla_server_module.deserialize_image_payload(image)
    assert parsed.mode == "RGB"
    assert parsed.size == (6, 6)


def test_deserialize_image_payload_from_1d_encoded_bytes(xvla_server_module, valid_payload_png_bytes):
    parsed = xvla_server_module.deserialize_image_payload(valid_payload_png_bytes)
    assert parsed.mode == "RGB"
    assert parsed.size == (8, 8)


def test_deserialize_image_payload_rejects_invalid_shape(xvla_server_module):
    with pytest.raises(ValueError, match="2D or 3D"):
        xvla_server_module.deserialize_image_payload(np.zeros((2, 2, 2, 2), dtype=np.uint8))


def test_get_instruction_accepts_primary_and_fallback_keys(xvla_server_module):
    assert xvla_server_module.get_instruction({"instruction": "pick"}) == "pick"
    assert xvla_server_module.get_instruction({"language_instruction": "place"}) == "place"


def test_get_instruction_rejects_invalid_input(xvla_server_module):
    with pytest.raises(ValueError, match="Missing field"):
        xvla_server_module.get_instruction({})
    with pytest.raises(ValueError, match="non-empty"):
        xvla_server_module.get_instruction({"instruction": "   "})


def test_get_primary_image_prefers_image_key_order(xvla_server_module):
    payload = {
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "image0": np.ones((2, 2, 3), dtype=np.uint8),
    }
    parsed = xvla_server_module.get_primary_image(payload)
    assert parsed.size == (2, 2)


def test_predict_action_returns_400_for_payload_errors(xvla_server_module):
    response = xvla_server_module.predict_action({"image": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert response.status_code == 400
    assert "Missing field" in _response_json(response)["error"]


def test_predict_action_returns_500_for_internal_errors(xvla_server_module, monkeypatch):
    xvla_server_module.cfg = SimpleNamespace(use_film=False)
    xvla_server_module.model = object()
    xvla_server_module.processor = object()
    xvla_server_module.action_head = object()
    xvla_server_module.proprio_projector = object()
    xvla_server_module.expected_proprio_dim = 10

    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(xvla_server_module, "get_vla_action", _boom)

    response = xvla_server_module.predict_action(
        {"instruction": "pick", "image": np.zeros((4, 4, 3), dtype=np.uint8)}
    )
    assert response.status_code == 500
    assert _response_json(response)["error"] == "Internal server error"


def test_predict_action_success_shape(xvla_server_module):
    xvla_server_module.cfg = SimpleNamespace(use_film=False)
    xvla_server_module.model = object()
    xvla_server_module.processor = object()
    xvla_server_module.action_head = object()
    xvla_server_module.proprio_projector = object()
    xvla_server_module.expected_proprio_dim = 10

    response = xvla_server_module.predict_action(
        {"instruction": "pick", "image": np.zeros((4, 4, 3), dtype=np.uint8)}
    )
    payload = _response_json(response)
    assert response.status_code == 200
    assert "action" in payload
    assert "action_chunk" in payload
    assert len(payload["action"]) == 8
