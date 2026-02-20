#!/bin/bash
# Phase 2: Start airgapped Surya OCR service
# Run this AFTER Phase 1 model download is complete

set -e

echo "========================================"
echo "PHASE 2: Starting Airgapped Surya OCR"
echo "========================================"
echo ""

# Check if llamaindex_internal network exists
if ! docker network inspect llamaindex_internal > /dev/null 2>&1; then
    echo "Creating llamaindex_internal network..."
    docker network create llamaindex_internal
else
    echo "llamaindex_internal network exists"
fi

# Check if models were downloaded (by checking volume)
echo "Checking for downloaded models..."
if docker volume inspect suryaocr_surya_models > /dev/null 2>&1; then
    echo "Model volume found"
else
    echo ""
    echo "WARNING: Model volume not found!"
    echo "Please run Phase 1 first: ./setup-phase1-download.sh"
    echo ""
    exit 1
fi

# Start the airgapped service
echo ""
echo "Starting Surya OCR service in airgapped mode..."
echo "The container will ONLY be connected to llamaindex_internal network"
echo ""

docker-compose up -d

echo ""
echo "========================================"
echo "Phase 2 Complete!"
echo "========================================"
echo ""
echo "Surya OCR is now running in airgapped mode."
echo ""
echo "Service endpoints:"
echo "  - REST API:    http://localhost:8080"
echo "  - API Docs:    http://localhost:8080/docs"
echo "  - Streamlit:   http://localhost:8501 (if using streamlit command)"
echo ""
echo "Security:"
echo "  - Connected only to: llamaindex_internal (no internet access)"
echo "  - TRANSFORMERS_OFFLINE=1"
echo "  - HF_HUB_OFFLINE=1"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f      # View logs"
echo "  docker-compose restart      # Restart service"
echo "  docker-compose down         # Stop service"
echo ""



