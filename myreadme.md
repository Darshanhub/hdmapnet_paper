# This is for local build
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install -r requirements.txt
./demo_cpu.sh --dataroot dataset/Copy_of_images/images \
              --image_dir dataset/Copy_of_images/images \
              --modelf artifacts/checkpoints/model29.pt \
              --limit 5 \
              --output_dir outputs/demo_cpu_images

# This is for docker run
## Building
docker build -t hdmapnet:latest .

## Running
docker run --rm \
  -v "$PWD/dataset/Copy_of_images/images:/data" \
  -v "$PWD/artifacts/checkpoints:/checkpoints" \
  -v "$PWD/outputs/docker_images:/outputs" \
  -e MODEL_PATH=/checkpoints/model29.pt \
  -e IMAGE_DIR=/data \
  hdmapnet:latest \
  --limit 10