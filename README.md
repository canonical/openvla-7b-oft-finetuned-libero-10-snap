# openvla-oft-finetuned-libero-10

Inference snap for **OpenVLA-OFT finetuned on LIBERO-10** with two serving interfaces:

- FastAPI multipart (`generic-cpu-fastapi`, `nvidia-gpu-amd64-fastapi`)
- XVLA JSON (`generic-cpu-xvla`, `nvidia-gpu-amd64-xvla`)

The snap package name in `snap/snapcraft.yaml` is:

- `openvla-oft-finetuned-libero-10`

## Build From Source

Clone the repository:

```shell
git clone https://github.com/canonical/openvla-7b-oft-finetuned-libero-10-snap.git
cd openvla-7b-oft-finetuned-libero-10-snap
```

```bash
snapcraft pack -v
```

Install the generated snap and components:

```bash
sudo snap install --dangerous ./openvla-oft-finetuned-libero-10_*.snap
sudo snap install --dangerous ./openvla-oft-finetuned-libero-10+*.comp
```

## Select Engine

Use the snap CLI app (`openvla-oft-finetuned-libero-10`) to inspect and select runtime engines:

```bash
sudo openvla-oft-finetuned-libero-10 list-engines
sudo openvla-oft-finetuned-libero-10 use-engine --auto
```

Or choose explicitly:

```bash
sudo openvla-oft-finetuned-libero-10 use-engine generic-cpu-xvla
sudo openvla-oft-finetuned-libero-10 use-engine nvidia-gpu-amd64-xvla
sudo openvla-oft-finetuned-libero-10 use-engine generic-cpu-fastapi
sudo openvla-oft-finetuned-libero-10 use-engine nvidia-gpu-amd64-fastapi
```

## Start And Check Server

```bash
sudo snap start openvla-oft-finetuned-libero-10.server
sudo snap services openvla-oft-finetuned-libero-10
sudo snap logs -f openvla-oft-finetuned-libero-10.server
```

Default serving port is managed by modelctl config (`http.port`, usually `9090`).

## XVLA Runtime Configuration

For XVLA engines, model behavior can be configured via `modelctl`:

```bash
sudo openvla-oft-finetuned-libero-10 set model.unnorm_key=""
sudo openvla-oft-finetuned-libero-10 set model.num_images="1"
sudo openvla-oft-finetuned-libero-10 set model.use_proprio="true"
sudo openvla-oft-finetuned-libero-10 set model.proprio_dim="10"
sudo openvla-oft-finetuned-libero-10 set model.user_config=""
```

Optional JSON user config can override these values (when set in `model.user_config`).
Supported JSON shapes are either flat or nested under `model`:

```json
{
  "model": {
    "unnorm_key": "libero_task",
    "num_images": 1,
    "use_proprio": true,
    "proprio_dim": 10
  }
}
```

Precedence for XVLA is:

1. Built-in defaults
2. `modelctl` values
3. User JSON config (`model.user_config`) if present

## Call The API

### XVLA JSON Endpoint

Use the included test client:

```bash
python3 test_xvla.py \
  --url http://localhost:9090/act \
  --image /path/to/image.png \
  --prompt "pick up the blue block"
```

Or send JSON directly (payload uses `instruction`, `image`).

### FastAPI Multipart Endpoint

When using a `*-fastapi` engine:

```bash
curl -X POST "http://localhost:9090/act" \
  -F "text=pick up the blue block" \
  -F "image=@/path/to/image.png"
```

## Resources

- [Inference Snaps documentation](https://documentation.ubuntu.com/inference-snaps/)

## Unit Tests

Run Python unit tests for XVLA server logic:

```bash
pytest tests/unit -v
```
