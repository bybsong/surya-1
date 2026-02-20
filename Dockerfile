# Dockerfile for Surya OCR v0.17.0 with RTX 5090 support
# Uses PyTorch 2.7.0 with CUDA 12.8 for compatibility with RTX 5090 (Blackwell architecture, compute capability 12.0)

FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set environment variables to prevent interactive prompts and optimize for RTX 5090
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_DEVICE=cuda
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Security: Disable telemetry - these will be set more restrictively after model download
ENV HUGGINGFACE_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV DISABLE_MLFLOW_INTEGRATION=true

# Model cache directory - persisted via volume mount
ENV MODEL_CACHE_DIR=/root/.cache/datalab/models

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA installation and RTX 5090 compatibility
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Copy requirements and install Python dependencies
COPY pyproject.toml poetry.lock* ./

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry==1.8.3

# Configure poetry to not create virtual environment (we're already in container)
RUN poetry config virtualenvs.create false

# Install dependencies (without dev dependencies, without installing the package itself yet)
RUN poetry install --only main --no-root

# Install additional runtime dependencies
RUN pip install --no-cache-dir \
    streamlit \
    fastapi \
    uvicorn \
    python-multipart

# Copy the entire application
COPY . .

# Install the application in editable mode
RUN poetry install --only main

# CRITICAL: Reinstall compatible torch/torchvision AFTER poetry install
# Poetry upgrades torch to 2.8.0 which breaks torchvision compatibility
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.0+cu128 \
    torchvision==0.22.0+cu128 \
    torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Final verification of torch/torchvision compatibility
RUN python -c "import torch; import torchvision; import transformers; from transformers import PreTrainedModel; print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}, transformers: {transformers.__version__}')"

# Copy and set up the entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create directories for data, results, and models
RUN mkdir -p /app/data /app/results /app/surya-output /root/.cache/datalab/models

# Set permissions on scripts
RUN chmod +x /app/*.py 2>/dev/null || true

# Expose ports for different applications
# 8501 - Streamlit web UI
# 8080 - FastAPI REST API
# 7860 - Gradio port (if needed)
EXPOSE 8501 8080 7860

# Health check to verify GPU access
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU Health Check Passed')"

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command - run FastAPI app (can be overridden)
CMD ["python", "-m", "uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8080"]

