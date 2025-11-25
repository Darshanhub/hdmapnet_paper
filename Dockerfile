# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch-scatter==2.1.2 \
        -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

COPY . .

ENV MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p "$MPLCONFIGDIR"

ENTRYPOINT ["bash", "scripts/docker_entrypoint.sh"]
