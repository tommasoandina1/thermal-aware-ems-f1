FROM python:3.11-slim

WORKDIR /app

# Build toolchain. Most scientific wheels are prebuilt, but fastf1 pulls in a
# few packages that still compile from source on some platforms.
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# --- PyTorch, CPU-only -------------------------------------------------------
# Installed on its own, before everything else, for two reasons:
#   1. --index-url REPLACES PyPI rather than adding to it, so this cannot be a
#      line in requirements.txt: pip would then look for fastf1 and jupyter on
#      download.pytorch.org and fail.
#   2. the default PyPI wheel on x86_64 is the CUDA build: ~1.2 GB of torch
#      plus ~2.7 GB of nvidia-* runtime packages, for two 256x256 MLPs.
#
# On Apple Silicon (linux/arm64) there is no CUDA build at all, so plain PyPI
# is already CPU-only; if this line fails on an arm64 build, replace it with
#     RUN pip install --no-cache-dir torch
# To train on a GPU machine later, swap /whl/cpu for /whl/cu121.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# --- everything else ---------------------------------------------------------
COPY requirements.txt requirements-rl.txt requirements-dev.txt ./
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements-rl.txt \
    -r requirements-dev.txt

# FastF1 writes its cache here instead of into the repository. The bind mount
# in docker-compose.yaml maps the project root to /app, so a cache inside the
# project would land back on the host and end up in git - which is how the
# 61 MB scripts/f1_cache/ directory got into the history in the first place.
ENV FASTF1_CACHE=/cache
RUN mkdir -p /cache

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
