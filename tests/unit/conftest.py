import argparse
import contextlib
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
XVLA_SERVER_PATH = REPO_ROOT / "components" / "xvla-server-interface" / "server.py"


def _install_stub_modules():
    json_numpy = types.ModuleType("json_numpy")
    json_numpy.patch = lambda: None
    json_numpy.loads = lambda s: np.asarray(json.loads(s))
    json_numpy.dumps = lambda arr: json.dumps(np.asarray(arr).tolist())
    sys.modules["json_numpy"] = json_numpy

    torch = types.ModuleType("torch")

    @contextlib.contextmanager
    def inference_mode():
        yield

    torch.inference_mode = inference_mode
    sys.modules["torch"] = torch

    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None
    sys.modules["uvicorn"] = uvicorn

    responses = types.ModuleType("fastapi.responses")

    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    responses.JSONResponse = JSONResponse
    sys.modules["fastapi.responses"] = responses

    fastapi = types.ModuleType("fastapi")

    class Response:
        def __init__(self, status_code=200, content="", media_type="application/json"):
            self.status_code = status_code
            self.content = content
            self.media_type = media_type

    class FastAPI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get(self, _path):
            def decorator(func):
                return func

            return decorator

        def post(self, _path):
            def decorator(func):
                return func

            return decorator

    fastapi.FastAPI = FastAPI
    fastapi.Response = Response
    sys.modules["fastapi"] = fastapi

    openvla_utils = types.ModuleType("experiments.robot.openvla_utils")

    class _DummyModel:
        def __init__(self):
            self.norm_stats = {"default": {"mean": 0.0, "std": 1.0}}
            self.llm_dim = 16

        def to(self, _device):
            return self

        def eval(self):
            return self

    class _DummyProjector:
        def to(self, _device):
            return self

    openvla_utils.get_vla = lambda _cfg: _DummyModel()
    openvla_utils.get_processor = lambda _cfg: object()
    openvla_utils.get_action_head = lambda _cfg, _llm_dim: _DummyProjector()
    openvla_utils.get_proprio_projector = lambda _cfg, llm_dim, proprio_dim: _DummyProjector()
    openvla_utils.get_vla_action = lambda *args, **kwargs: np.zeros((1, 8), dtype=np.float32)

    sys.modules["experiments"] = types.ModuleType("experiments")
    sys.modules["experiments.robot"] = types.ModuleType("experiments.robot")
    sys.modules["experiments.robot.openvla_utils"] = openvla_utils

    constants = types.ModuleType("prismatic.vla.constants")
    constants.NUM_ACTIONS_CHUNK = 1
    constants.PROPRIO_DIM = 10
    sys.modules["prismatic"] = types.ModuleType("prismatic")
    sys.modules["prismatic.vla"] = types.ModuleType("prismatic.vla")
    sys.modules["prismatic.vla.constants"] = constants


@pytest.fixture
def xvla_server_module(monkeypatch):
    _install_stub_modules()

    monkeypatch.setenv("PYTEST_RUNNING", "1")
    original_argv = sys.argv[:]
    sys.argv = ["xvla-server-test"]

    spec = importlib.util.spec_from_file_location("xvla_server_under_test", XVLA_SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sys.argv = original_argv
    return module


@pytest.fixture
def valid_payload_png_bytes():
    from PIL import Image
    import io

    image = Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


@pytest.fixture
def namespace_defaults(tmp_path):
    return argparse.Namespace(
        unnorm_key="",
        num_images=1,
        use_proprio=True,
        proprio_dim=10,
        user_config=str(tmp_path / "missing.json"),
    )
