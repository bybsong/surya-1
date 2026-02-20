#!/usr/bin/env python3
"""
FastAPI REST API for Surya OCR v0.17.0
Provides endpoints for OCR, detection, layout analysis, and table recognition

Designed for airgapped operation on llamaindex_internal network
"""

import os
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from PIL import Image
import pypdfium2

from surya.models import load_predictors
from surya.settings import settings
from surya.common.surya.schema import TaskNames

# Initialize FastAPI app
app = FastAPI(
    title="Surya OCR API",
    description="REST API for Surya OCR v0.17.0 with RTX 5090 optimization (Airgapped)",
    version="0.17.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global predictors - loaded once at startup
predictors = None

@app.on_event("startup")
async def startup_event():
    """Load Surya predictors once at startup"""
    global predictors
    print("Loading Surya predictors...")
    print(f"Torch device: {settings.TORCH_DEVICE_MODEL}")
    predictors = load_predictors()
    print(f"Loaded predictors: {list(predictors.keys())}")
    print("[OK] Surya predictors loaded successfully")

# Output directory
OUTPUT_DIR = Path(os.environ.get("SURYA_OUTPUT_DIR", "/app/surya-output"))
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Surya OCR API is running",
        "version": "0.17.0",
        "status": "healthy",
        "airgapped": os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    import torch
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "device_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        }
    else:
        gpu_info = {"cuda_available": False}
    
    return {
        "status": "healthy",
        "version": "0.17.0",
        "predictors_loaded": list(predictors.keys()) if predictors else [],
        "torch_device": str(settings.TORCH_DEVICE_MODEL),
        "output_directory": str(OUTPUT_DIR),
        "airgapped": os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
        "gpu_info": gpu_info
    }

def save_results(results, filename_base: str, endpoint: str):
    """Save results to output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{filename_base}_{endpoint}_{timestamp}.json"
    
    # Convert results to serializable format
    if hasattr(results[0], 'model_dump'):
        serializable_results = [r.model_dump() for r in results]
    else:
        serializable_results = results
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    return str(output_file)

def load_image_from_upload(file_content: bytes, content_type: str, page_number: int = 1) -> tuple:
    """Load image from uploaded file (handles PDF and images)"""
    if content_type == "application/pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            
            doc = pypdfium2.PdfDocument(tmp_file.name)
            if page_number > len(doc):
                raise HTTPException(status_code=400, detail=f"Page {page_number} not found. PDF has {len(doc)} pages.")
            
            page = doc[page_number - 1]
            pil_image = page.render(scale=settings.IMAGE_DPI/72).to_pil()
            pil_image_highres = page.render(scale=settings.IMAGE_DPI_HIGHRES/72).to_pil()
            doc.close()
            
            os.unlink(tmp_file.name)
            return pil_image, pil_image_highres
    else:
        pil_image = Image.open(io.BytesIO(file_content)).convert("RGB")
        return pil_image, pil_image

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    page_number: Optional[int] = Form(1),
    recognize_math: bool = Form(True),
    with_bboxes: bool = Form(True),
    return_words: bool = Form(True)
):
    """
    Full OCR processing with text detection and recognition.
    
    - **file**: Image or PDF file to process
    - **page_number**: Page number (1-indexed) for PDFs
    - **recognize_math**: Enable math formula recognition
    - **with_bboxes**: Return bounding boxes for text
    - **return_words**: Return word-level segmentation
    """
    try:
        file_content = await file.read()
        filename_base = Path(file.filename).stem
        
        pil_image, pil_image_highres = load_image_from_upload(
            file_content, file.content_type, page_number
        )
        
        # Choose task based on bbox preference
        task = TaskNames.ocr_with_boxes if with_bboxes else TaskNames.ocr_without_boxes
        
        # Run OCR
        results = predictors["recognition"](
            [pil_image],
            task_names=[task],
            det_predictor=predictors["detection"],
            highres_images=[pil_image_highres],
            math_mode=recognize_math,
            return_words=return_words
        )
        
        # Save results
        output_file = save_results(results, filename_base, "ocr")
        
        return {
            "status": "success",
            "results": [r.model_dump() for r in results],
            "output_file": output_file,
            "processed_file": file.filename,
            "page_number": page_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr-all-pages")
async def ocr_all_pages_endpoint(
    file: UploadFile = File(...),
    recognize_math: bool = Form(True),
    batch_size: int = Form(8)
):
    """
    Process all pages of a PDF with batch processing.
    Results are saved to individual JSON files per page.
    
    - **file**: PDF file to process
    - **recognize_math**: Enable math formula recognition
    - **batch_size**: Number of pages to process in each batch
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="This endpoint only supports PDF files")
        
        file_content = await file.read()
        filename_base = Path(file.filename).stem
        
        # Create output folder for this document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_output_dir = OUTPUT_DIR / f"{filename_base}_{timestamp}"
        doc_output_dir.mkdir(exist_ok=True)
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            
            doc = pypdfium2.PdfDocument(tmp_file.name)
            total_pages = len(doc)
            
            all_results = []
            processed_pages = 0
            
            # Process pages in batches
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                current_batch_size = batch_end - batch_start
                
                print(f"Processing pages {batch_start + 1}-{batch_end} of {total_pages}...")
                
                # Prepare batch of images
                batch_images = []
                batch_images_highres = []
                page_numbers = []
                
                for page_idx in range(batch_start, batch_end):
                    page = doc[page_idx]
                    pil_image = page.render(scale=settings.IMAGE_DPI/72).to_pil()
                    pil_image_highres = page.render(scale=settings.IMAGE_DPI_HIGHRES/72).to_pil()
                    
                    batch_images.append(pil_image)
                    batch_images_highres.append(pil_image_highres)
                    page_numbers.append(page_idx + 1)
                
                # Process batch with Surya
                batch_results = predictors["recognition"](
                    batch_images,
                    task_names=[TaskNames.ocr_with_boxes] * current_batch_size,
                    det_predictor=predictors["detection"],
                    highres_images=batch_images_highres,
                    math_mode=recognize_math,
                    return_words=True
                )
                
                # Save individual page results
                for i, (result, page_num) in enumerate(zip(batch_results, page_numbers)):
                    page_file = doc_output_dir / f"page_{page_num:03d}.json"
                    
                    page_data = {
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "ocr_result": result.model_dump(),
                        "processed_at": datetime.now().isoformat()
                    }
                    
                    with open(page_file, 'w', encoding='utf-8') as f:
                        json.dump(page_data, f, indent=2, ensure_ascii=False)
                    
                    all_results.append({"page": page_num, "file": str(page_file)})
                
                processed_pages += current_batch_size
            
            doc.close()
            os.unlink(tmp_file.name)
        
        # Save summary file
        summary_file = doc_output_dir / "summary.json"
        summary_data = {
            "document_name": file.filename,
            "total_pages": total_pages,
            "processed_pages": processed_pages,
            "batch_size_used": batch_size,
            "output_directory": str(doc_output_dir),
            "processed_at": datetime.now().isoformat(),
            "math_recognition": recognize_math
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": f"Processed {processed_pages} pages successfully",
            "total_pages": total_pages,
            "output_directory": str(doc_output_dir),
            "summary_file": str(summary_file),
            "page_files": all_results,
            "processed_file": file.filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"All-pages OCR processing failed: {str(e)}")

@app.post("/detection")
async def detection_endpoint(
    file: UploadFile = File(...),
    page_number: Optional[int] = Form(1)
):
    """
    Text detection only - finds text regions without OCR.
    
    - **file**: Image or PDF file to process
    - **page_number**: Page number (1-indexed) for PDFs
    """
    try:
        file_content = await file.read()
        filename_base = Path(file.filename).stem
        
        pil_image, _ = load_image_from_upload(file_content, file.content_type, page_number)
        
        # Run detection
        results = predictors["detection"]([pil_image])
        
        # Save results
        output_file = save_results(results, filename_base, "detection")
        
        return {
            "status": "success",
            "results": [r.model_dump() for r in results],
            "output_file": output_file,
            "processed_file": file.filename,
            "page_number": page_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/layout")
async def layout_endpoint(
    file: UploadFile = File(...),
    page_number: Optional[int] = Form(1)
):
    """
    Layout analysis - detects document structure (headers, paragraphs, tables, etc.)
    
    - **file**: Image or PDF file to process
    - **page_number**: Page number (1-indexed) for PDFs
    """
    try:
        file_content = await file.read()
        filename_base = Path(file.filename).stem
        
        pil_image, _ = load_image_from_upload(file_content, file.content_type, page_number)
        
        # Run layout analysis
        results = predictors["layout"]([pil_image])
        
        # Save results
        output_file = save_results(results, filename_base, "layout")
        
        return {
            "status": "success",
            "results": [r.model_dump() for r in results],
            "output_file": output_file,
            "processed_file": file.filename,
            "page_number": page_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layout analysis failed: {str(e)}")

@app.post("/table-rec")
async def table_rec_endpoint(
    file: UploadFile = File(...),
    page_number: Optional[int] = Form(1)
):
    """
    Table recognition - detects and structures tables in documents.
    
    - **file**: Image or PDF file to process
    - **page_number**: Page number (1-indexed) for PDFs
    """
    try:
        file_content = await file.read()
        filename_base = Path(file.filename).stem
        
        pil_image, _ = load_image_from_upload(file_content, file.content_type, page_number)
        
        # Run table recognition
        results = predictors["table_rec"]([pil_image])
        
        # Save results
        output_file = save_results(results, filename_base, "table_rec")
        
        return {
            "status": "success",
            "results": [r.model_dump() for r in results],
            "output_file": output_file,
            "processed_file": file.filename,
            "page_number": page_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Table recognition failed: {str(e)}")

@app.post("/ocr-error")
async def ocr_error_endpoint(
    file: UploadFile = File(...),
    page_number: Optional[int] = Form(1)
):
    """
    OCR error detection - identifies potential OCR errors.
    
    - **file**: Image or PDF file to process
    - **page_number**: Page number (1-indexed) for PDFs
    """
    try:
        file_content = await file.read()
        filename_base = Path(file.filename).stem
        
        pil_image, _ = load_image_from_upload(file_content, file.content_type, page_number)
        
        # Run OCR error detection
        results = predictors["ocr_error"]([pil_image])
        
        # Save results
        output_file = save_results(results, filename_base, "ocr_error")
        
        return {
            "status": "success",
            "results": [r.model_dump() for r in results],
            "output_file": output_file,
            "processed_file": file.filename,
            "page_number": page_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR error detection failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)



