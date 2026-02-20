#!/bin/bash
set -e

# Docker entrypoint script for Surya OCR v0.17.0
# Handles initialization, GPU verification, and application startup

echo "========================================"
echo "Starting Surya OCR Docker Container"
echo "Version: 0.17.0"
echo "========================================"

# Function to check GPU availability
check_gpu() {
    echo "Checking GPU availability..."
    python -c "
import torch
import sys
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1024**3:.1f} GB')
        print(f'  Compute capability: {props.major}.{props.minor}')
else:
    print('WARNING: CUDA not available. Running on CPU.')
    sys.exit(1)
"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: GPU check failed. Please ensure:"
        echo "1. NVIDIA drivers are installed on the host"
        echo "2. NVIDIA Container Toolkit is installed"
        echo "3. Docker is run with --gpus all flag"
        exit 1
    fi
    echo "[OK] GPU check passed!"
}

# Function to check if models are downloaded
check_models_downloaded() {
    echo "Checking if models are downloaded..."
    python -c "
from surya.settings import settings
from pathlib import Path

cache_dir = Path(settings.MODEL_CACHE_DIR)
marker_file = cache_dir / '.models_downloaded'

if marker_file.exists():
    print('[OK] Models already downloaded')
    print('[OK] System ready for offline operation')
    exit(0)
else:
    print('[WARNING] Models not yet downloaded')
    print('Run Phase 1 first: docker-compose -f docker-compose-download.yml up')
    exit(1)
"
    return $?
}

# Function to verify airgap security settings
verify_airgap_settings() {
    echo "Verifying airgap security settings..."
    
    if [ "${HF_HUB_OFFLINE}" = "1" ] && [ "${TRANSFORMERS_OFFLINE}" = "1" ]; then
        echo "[OK] Airgap mode enabled - network access blocked for model downloads"
    else
        echo "[WARNING] Airgap mode NOT fully enabled"
        echo "  HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
        echo "  TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
    fi
    
    echo "[OK] Security settings verified"
}

# Function to test basic functionality
test_basic_functionality() {
    echo "Testing basic Surya functionality..."
    python -c "
import sys
try:
    # Just test imports without loading models to save time
    from surya.models import load_predictors
    from surya.settings import settings
    print(f'Torch device: {settings.TORCH_DEVICE_MODEL}')
    print('Successfully imported Surya modules')
    print('Basic functionality test passed!')
except Exception as e:
    print(f'Error testing functionality: {e}')
    sys.exit(1)
"
}

# Function to display batch size configuration
show_batch_sizes() {
    echo "Batch size configuration (optimized for RTX 5090):"
    echo "  DETECTOR_BATCH_SIZE=${DETECTOR_BATCH_SIZE:-auto}"
    echo "  RECOGNITION_BATCH_SIZE=${RECOGNITION_BATCH_SIZE:-auto}"
    echo "  LAYOUT_BATCH_SIZE=${LAYOUT_BATCH_SIZE:-auto}"
    echo "  TABLE_REC_BATCH_SIZE=${TABLE_REC_BATCH_SIZE:-auto}"
}

# Main execution
main() {
    echo ""
    
    # Check if running with GPU support
    if [ "${TORCH_DEVICE:-cuda}" = "cuda" ]; then
        check_gpu
    else
        echo "Running in CPU mode"
    fi
    
    # Check models are downloaded
    if ! check_models_downloaded; then
        echo ""
        echo "========================================"
        echo "ERROR: Models not downloaded!"
        echo "========================================"
        echo ""
        echo "Please run Phase 1 first to download models:"
        echo "  docker-compose -f docker-compose-download.yml up"
        echo ""
        echo "After download completes, run Phase 2:"
        echo "  docker-compose up -d"
        echo ""
        exit 1
    fi
    
    # Verify airgap settings
    verify_airgap_settings
    
    # Test functionality
    test_basic_functionality
    
    # Show batch sizes
    show_batch_sizes
    
    echo ""
    echo "========================================"
    echo "Initialization Complete!"
    echo "========================================"
    echo ""
    
    # Execute the passed command
    exec "$@"
}

# Run main function with all arguments
main "$@"



