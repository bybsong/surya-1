#!/bin/bash
# Test script for Surya OCR API
# Run this after the service is started to verify it's working

set -e

API_URL="${SURYA_API_URL:-http://localhost:8080}"

echo "========================================"
echo "Testing Surya OCR API"
echo "========================================"
echo ""
echo "API URL: $API_URL"
echo ""

# Health check
echo "1. Health Check..."
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

# Root endpoint
echo "2. Root Endpoint..."
curl -s "$API_URL/" | python3 -m json.tool
echo ""

# Check if test image exists
if [ -f "tests/assets/test_latex.png" ]; then
    echo "3. Testing OCR endpoint with test image..."
    curl -X POST "$API_URL/ocr" \
        -F "file=@tests/assets/test_latex.png" \
        -F "recognize_math=true" | python3 -m json.tool | head -50
    echo "..."
    echo ""
else
    echo "3. Skipping OCR test (no test image found at tests/assets/test_latex.png)"
fi

echo ""
echo "========================================"
echo "API Test Complete!"
echo "========================================"
echo ""
echo "To test with your own files:"
echo "  curl -X POST '$API_URL/ocr' -F 'file=@your-image.png'"
echo "  curl -X POST '$API_URL/ocr-all-pages' -F 'file=@your-document.pdf'"
echo ""



