# Surya OCR Docker Setup (v0.17.0)

Two-phase deployment with airgapping for security.

## Requirements

- Docker Desktop with WSL2 backend
- NVIDIA GPU with Docker GPU support (nvidia-container-toolkit)
- RTX 5090 (or compatible GPU with CUDA 12.8+ support)
- `llamaindex_internal` Docker network (created automatically)

## Quick Start (WSL)

```bash
cd /mnt/c/Users/bybso/suryaOCR

# Make scripts executable
chmod +x *.sh

# Phase 1: Build and download models (requires internet)
./setup-phase1-download.sh

# Phase 2: Start airgapped service (no internet)
./setup-phase2-airgap.sh

# Test the API
./test-api.sh
```

## Two-Phase Deployment

### Phase 1: Model Download (WITH Network)

Downloads all required models (~10-15 GB) while network access is available.

```bash
# Build the Docker image
docker build -t surya-ocr:0.17.0 .

# Run model download container
docker-compose -f docker-compose-download.yml up
```

Models are saved to Docker volumes:
- `surya_models` - Surya model cache
- `surya_huggingface` - HuggingFace cache

### Phase 2: Airgapped Production (NO Network)

After models are downloaded, run the service with network restrictions:

```bash
# Ensure llamaindex_internal network exists
docker network create llamaindex_internal 2>/dev/null || true

# Start the airgapped service
docker-compose up -d
```

## Network Security

The production container:
- Connects ONLY to `llamaindex_internal` network
- Has NO internet access
- Sets `TRANSFORMERS_OFFLINE=1`
- Sets `HF_HUB_OFFLINE=1`
- Sets `HF_DATASETS_OFFLINE=1`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/docs` | GET | Interactive API documentation |
| `/ocr` | POST | Full OCR (detection + recognition) |
| `/ocr-all-pages` | POST | Process all pages of a PDF |
| `/detection` | POST | Text detection only |
| `/layout` | POST | Document layout analysis |
| `/table-rec` | POST | Table structure recognition |
| `/ocr-error` | POST | OCR error detection |

## Example API Usage

```bash
# Single page OCR
curl -X POST http://localhost:8080/ocr \
    -F "file=@document.png" \
    -F "recognize_math=true"

# All pages PDF OCR
curl -X POST http://localhost:8080/ocr-all-pages \
    -F "file=@document.pdf" \
    -F "batch_size=8"

# Layout analysis
curl -X POST http://localhost:8080/layout \
    -F "file=@document.png"
```

## Output Directory

OCR results are saved to:
- Container: `/app/surya-output`
- Host mount: `C:\Users\bybso\Sync\PMHx\SuryaOut` (Windows)

## RTX 5090 Optimization

Batch sizes are optimized for 32GB VRAM:
- `DETECTOR_BATCH_SIZE=64`
- `RECOGNITION_BATCH_SIZE=1024`
- `LAYOUT_BATCH_SIZE=64`
- `TABLE_REC_BATCH_SIZE=128`

## Troubleshooting

### Models not downloading
- Check internet connection
- Verify GPU is accessible: `docker run --gpus all nvidia/cuda:12.8.0-base nvidia-smi`

### Container fails to start
- Ensure Phase 1 completed successfully
- Check for offline marker: `docker exec surya-ocr cat /root/.cache/datalab/models/.models_downloaded`

### GPU not detected
- Install nvidia-container-toolkit
- Restart Docker Desktop
- Verify with: `docker info | grep -i gpu`

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container build definition |
| `docker-compose.yml` | Phase 2: Airgapped production |
| `docker-compose-download.yml` | Phase 1: Model download |
| `docker-entrypoint.sh` | Container initialization |
| `download-models.py` | Model download script |
| `api_app.py` | FastAPI REST application |
| `build.sh` | Build helper script |
| `setup-phase1-download.sh` | Phase 1 automation |
| `setup-phase2-airgap.sh` | Phase 2 automation |
| `test-api.sh` | API test script |



