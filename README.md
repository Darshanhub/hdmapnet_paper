# HDMapNet Inference Helper

This repository wraps the original [HDMapNet](https://github.com/Tsinghua-MARS-Lab/HDMapNet) implementation with a streamlined inference script (`scripts/run_inference.py`) that can consume either a full nuScenes layout or a plain folder of RGB frames such as `dataset/Copy_of_images/images`. The script produces submission-ready JSON as well as easy-to-read PNG overlays.

## Local (virtualenv) usage

1. **Create an environment** (Python 3.10 recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
   pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
   pip install -r requirements.txt
   ```

2. **Run inference** with either a nuScenes root (`dataset/nuScenes`) or an image directory (`dataset/Copy_of_images/images`). The helper script auto-detects plain folders, but you can also pass `--image_dir` explicitly:

   ```bash
   ./demo_cpu.sh \
     --dataroot dataset/Copy_of_images/images \
     --image_dir dataset/Copy_of_images/images \
     --modelf artifacts/checkpoints/model29.pt \
     --limit 5 \
     --output_dir outputs/demo_cpu_images
   ```

## Docker packaging

A self-contained Docker image is provided so you can deploy inference without recreating the Python toolchain on every machine.

### Build

```bash
docker build -t hdmapnet:latest .
```

### Required volumes & environment

| Mount / Env | Purpose | Example |
|-------------|---------|---------|
| `-v /path/to/dataset:/data` | nuScenes tree or folder of standalone images | `dataset/Copy_of_images/images` |
| `-v /path/to/checkpoints:/checkpoints` | directory containing `model.pt`/`model29.pt` | `artifacts/checkpoints` |
| `-v /path/to/output:/outputs` | host directory that will receive JSON/PNGs | `outputs/docker_run` |
| `-e MODEL_PATH=/checkpoints/model29.pt` | checkpoint inside the container | defaults to `/checkpoints/model.pt` |
| `-e IMAGE_DIR=/data` | (optional) force plain-image mode instead of nuScenes auto-detect | only needed when `/data` isn’t a nuScenes layout |
| `-e FORCE_CPU=0` | allow CUDA if you rebuild the image on a GPU base | default `1` |

### Run examples

Plain image folder (e.g., the provided `Copy_of_images` set):

```bash
docker run --rm \
  -v "$PWD/dataset/Copy_of_images/images:/data" \
  -v "$PWD/artifacts/checkpoints:/checkpoints" \
  -v "$PWD/outputs/docker_images:/outputs" \
  -e MODEL_PATH=/checkpoints/model29.pt \
  -e IMAGE_DIR=/data \
  hdmapnet:latest \
  --limit 10
```

nuScenes layout (mount the full dataset root that contains `samples/`, `sweeps/`, etc.):

```bash
docker run --rm \
  -v /path/to/nuscenes:/data \
  -v "$PWD/artifacts/checkpoints:/checkpoints" \
  -v "$PWD/outputs/docker_nusc:/outputs" \
  -e MODEL_PATH=/checkpoints/model29.pt \
  hdmapnet:latest \
  --version v1.0-mini \
  --limit 5
```

The entrypoint forwards any additional CLI switches (e.g., `--output_dir`, `--dry-run`, `--bsz 1`) directly to `scripts/run_inference.py`. Outputs appear under the mounted `/outputs` directory.

## Notes

- The Docker image is CPU-only by default. To leverage CUDA you would need to switch the base image (e.g., `nvidia/cuda`) and set `FORCE_CPU=0` when running the container. Torch 2.3.1 is already specified, so upgrading to a GPU variant only requires rebuilding with the appropriate PyTorch wheel index.
- `scripts/docker_entrypoint.sh` honours the same arguments as the local `demo_cpu.sh`, so once you validate a command locally you can copy the flags to the Docker invocation.
- All installs pin the same versions we verified locally, ensuring reproducible inference results.

## Google Colab workflow

Colab already provides Python 3.10 and CUDA-ready runtimes, so you only need to clone the repo, install the pinned requirements, and invoke the dedicated wrapper script.

1. **(Optional) enable GPU** – `Runtime > Change runtime type > Hardware accelerator > GPU`.
2. **Mount Drive** if your dataset/checkpoint lives there:

  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

3. **Clone & install** (runs once per session):

  ```bash
  !git clone https://github.com/YOUR-ORG/HDMapNet.git
  %cd HDMapNet
  !pip install --upgrade pip
  !pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
  !pip install -r requirements.txt
  !pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
  ```

4. **Run inference with the Colab helper** (`scripts/run_colab.py`). Supply the dataset (nuScenes root or plain folder) and checkpoint paths from your Colab filesystem or Drive:

  ```bash
  !python scripts/run_colab.py \
     --dataset /content/drive/MyDrive/hdmapnet_data/Copy_of_images/images \
     --image_dir /content/drive/MyDrive/hdmapnet_data/Copy_of_images/images \
     --model_path /content/drive/MyDrive/hdmapnet_data/model29.pt \
     --output_dir /content/hdmapnet_outputs \
     --limit 5 \
     --force-cpu  # drop this flag if you've enabled a GPU runtime
  ```

  For an official nuScenes layout, omit `--image_dir` and set `--dataset` to the folder that contains `samples/`, `sweeps/`, and the `v1.0-*` tables. You can pass any additional arguments supported by `scripts/run_inference.py` via the `--extra` flag, e.g. `--extra --bsz 2 --thickness 7`.

Outputs land in the provided `--output_dir` (e.g., `/content/hdmapnet_outputs`); you can zip that folder or copy it back to Drive when the run finishes. The helper enforces simple path checks so mis-typed Drive locations fail fast.
