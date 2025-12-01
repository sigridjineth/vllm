# MUVERA FDE Deployment Guide

This guide explains how to build and deploy the vLLM image with MUVERA FDE support.

## 1. Prerequisites

- Docker installed
- NVIDIA GPU with drivers installed (for running the container)

## 2. Build the Docker Image

The `Dockerfile.fde` is configured to:
1.  Use `nvidia/cuda:12.4.1-devel-ubuntu22.04` as the base image.
2.  Use `uv` to install Python 3.10 and manage the virtual environment.
3.  Install the modified vLLM source code from the local directory.
4.  **Automatically download and convert** the BGE-M3 model to an FDE-enabled model during the build process.

### Default Build (BAAI/bge-m3, ksim=6)

```bash
docker build -t vllm-muvera:latest -f Dockerfile.fde .
```

### Customizing FDE Parameters

You can override the default FDE parameters using Docker build arguments (`--build-arg`):

```bash
docker build -t vllm-muvera:custom -f Dockerfile.fde . \
  --build-arg MODEL_ID=BAAI/bge-m3 \
  --build-arg KSIM=6 \
  --build-arg D_PROJ=32 \
  --build-arg R_REPS=10 \
  --build-arg FILL_EMPTY=true
```

*   `MODEL_ID`: Source model ID (default: `BAAI/bge-m3`).
*   `KSIM`: SimHash bits (default: `6`).
*   `D_PROJ`: Projection dimension (default: `32`).
*   `R_REPS`: Number of repetitions (default: `10`).
*   `FILL_EMPTY`: Enable empty cluster filling (`true`/`false`, default: `true`).

## 3. Run the Container

Since the model is already embedded in the image at `/workspace/bge-m3-fde`, you don't need to mount any volumes for the model.

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  vllm-muvera:latest
```

*   The default command automatically serves the model at `/workspace/bge-m3-fde`.
*   You can still override the command if needed (e.g., to change `gpu-memory-utilization`).

## 4. Usage

### Generate Query Embedding (Sum Aggregation)

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/bge-m3-fde",
    "input": "What is MUVERA?",
    "is_document": false
  }'
```

### Generate Document Embedding (Mean Aggregation + Fill)

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/bge-m3-fde",
    "input": "MUVERA is a multi-vector retrieval approach...",
    "is_document": true
  }'
```
