#!/bin/bash
# Build script for Surya OCR Docker image
# Run this from WSL in the suryaOCR directory

set -e

echo "========================================"
echo "Building Surya OCR Docker Image v0.17.0"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo "Error: Dockerfile not found. Run this script from the suryaOCR directory."
    exit 1
fi

# Build the image
echo "Building Docker image..."
docker build -t surya-ocr:0.17.0 .

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download models (Phase 1):"
echo "   docker-compose -f docker-compose-download.yml up"
echo ""
echo "2. After download completes, start airgapped service (Phase 2):"
echo "   docker-compose up -d"
echo ""



