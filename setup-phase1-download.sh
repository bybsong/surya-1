#!/bin/bash
# Phase 1: Download models with network access
# Run this FIRST before starting the airgapped service

set -e

echo "========================================"
echo "PHASE 1: Model Download"
echo "========================================"
echo ""
echo "This will download all Surya models (~10-15 GB)"
echo "Internet access is required for this step only"
echo ""

# Check if image exists
if ! docker image inspect surya-ocr:0.17.0 > /dev/null 2>&1; then
    echo "Docker image not found. Building first..."
    ./build.sh
fi

# Ensure llamaindex_internal network exists (for later use)
echo "Ensuring llamaindex_internal network exists..."
docker network create llamaindex_internal 2>/dev/null || echo "Network already exists"

# Run the download container
echo ""
echo "Starting model download..."
echo "This may take 10-30 minutes depending on your internet connection."
echo ""

docker-compose -f docker-compose-download.yml up

echo ""
echo "========================================"
echo "Phase 1 Complete!"
echo "========================================"
echo ""
echo "Models have been downloaded to Docker volumes:"
echo "  - surya_models (Surya model cache)"
echo "  - surya_huggingface (HuggingFace cache)"
echo ""
echo "Next step - Start airgapped service (Phase 2):"
echo "  ./setup-phase2-airgap.sh"
echo ""



