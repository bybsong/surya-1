#!/usr/bin/env python3
"""
Model Download Script for Surya OCR v0.17.0
Downloads all required models before implementing airgap security restrictions

PHASE 1 SCRIPT - Run this BEFORE enabling network restrictions
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up environment for model downloads"""
    print("Setting up environment for model downloads...")
    
    # Ensure network access is enabled for downloads
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_DATASETS_OFFLINE'] = '0'
    
    # Disable telemetry but allow downloads
    os.environ['HUGGINGFACE_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'true'
    
    # Set device
    os.environ['TORCH_DEVICE'] = 'cuda' if os.environ.get('TORCH_DEVICE', 'cuda') == 'cuda' else 'cpu'
    
    print(f"Target device: {os.environ.get('TORCH_DEVICE', 'auto')}")

def download_detection_models():
    """Download text detection models"""
    print("\n" + "="*60)
    print("DOWNLOADING DETECTION MODELS")
    print("="*60)
    
    try:
        from surya.detection import DetectionPredictor
        print("Loading detection predictor...")
        predictor = DetectionPredictor()
        print("[OK] Detection models downloaded successfully")
        del predictor
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download detection models: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_foundation_models():
    """Download foundation models (shared by recognition and layout)"""
    print("\n" + "="*60)
    print("DOWNLOADING FOUNDATION MODELS")
    print("="*60)
    
    try:
        from surya.foundation import FoundationPredictor
        from surya.settings import settings
        
        # Download main recognition foundation model
        print(f"Loading foundation predictor (recognition checkpoint: {settings.RECOGNITION_MODEL_CHECKPOINT})...")
        foundation_predictor = FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
        print("[OK] Recognition foundation models downloaded")
        del foundation_predictor
        
        # Download layout foundation model (may be same or different checkpoint)
        print(f"Loading foundation predictor (layout checkpoint: {settings.LAYOUT_MODEL_CHECKPOINT})...")
        layout_foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        print("[OK] Layout foundation models downloaded")
        del layout_foundation
        
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download foundation models: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_recognition_models():
    """Download text recognition models"""
    print("\n" + "="*60)
    print("DOWNLOADING RECOGNITION MODELS")
    print("="*60)
    
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings
        
        print("Loading recognition predictor...")
        foundation = FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
        recognition_predictor = RecognitionPredictor(foundation)
        print("[OK] Recognition models downloaded successfully")
        del recognition_predictor, foundation
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download recognition models: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_layout_models():
    """Download layout analysis models"""
    print("\n" + "="*60)
    print("DOWNLOADING LAYOUT MODELS")
    print("="*60)
    
    try:
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings
        
        print("Loading layout predictor...")
        foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        predictor = LayoutPredictor(foundation)
        print("[OK] Layout models downloaded successfully")
        del predictor, foundation
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download layout models: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_table_rec_models():
    """Download table recognition models"""
    print("\n" + "="*60)
    print("DOWNLOADING TABLE RECOGNITION MODELS")
    print("="*60)
    
    try:
        from surya.table_rec import TableRecPredictor
        print("Loading table recognition predictor...")
        predictor = TableRecPredictor()
        print("[OK] Table recognition models downloaded successfully")
        del predictor
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download table recognition models: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_ocr_error_models():
    """Download OCR error detection models"""
    print("\n" + "="*60)
    print("DOWNLOADING OCR ERROR DETECTION MODELS")
    print("="*60)
    
    try:
        from surya.ocr_error import OCRErrorPredictor
        print("Loading OCR error predictor...")
        predictor = OCRErrorPredictor()
        print("[OK] OCR error detection models downloaded successfully")
        del predictor
        return True
    except Exception as e:
        print(f"[FAILED] Failed to download OCR error models: {e}")
        print("Note: OCR error detection is optional")
        import traceback
        traceback.print_exc()
        return True  # Don't fail the entire process for optional models

def test_model_loading():
    """Test that all models can be loaded successfully"""
    print("\n" + "="*60)
    print("TESTING FULL MODEL LOADING")
    print("="*60)
    
    try:
        from surya.models import load_predictors
        print("Loading all predictors for verification...")
        predictors = load_predictors()
        
        print("Available predictors:")
        for name in predictors.keys():
            print(f"  [OK] {name}")
        
        # Clean up to free GPU memory
        del predictors
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[OK] All models loaded successfully!")
        return True
    except Exception as e:
        print(f"[FAILED] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_cache():
    """Verify that models are properly cached"""
    print("\n" + "="*60)
    print("VERIFYING MODEL CACHE")
    print("="*60)
    
    from surya.settings import settings
    cache_dir = Path(settings.MODEL_CACHE_DIR)
    
    print(f"Model cache directory: {cache_dir}")
    
    if cache_dir.exists():
        cached_files = list(cache_dir.rglob("*"))
        model_files = [f for f in cached_files if f.is_file() and f.suffix in ['.bin', '.safetensors', '.json', '.txt', '.model']]
        
        print(f"Found {len(model_files)} model files in cache:")
        for f in sorted(model_files)[:15]:  # Show first 15
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.relative_to(cache_dir)} ({size_mb:.1f} MB)")
        
        if len(model_files) > 15:
            print(f"  ... and {len(model_files) - 15} more files")
        
        total_size = sum(f.stat().st_size for f in model_files) / (1024*1024*1024)
        print(f"Total cache size: {total_size:.2f} GB")
        
        if len(model_files) > 0:
            print("[OK] Models are properly cached")
            return True
        else:
            print("[FAILED] No model files found in cache")
            return False
    else:
        print("[FAILED] Model cache directory does not exist")
        return False

def create_offline_marker():
    """Create a marker file indicating models are downloaded and system can go offline"""
    print("\n" + "="*60)
    print("CREATING OFFLINE MARKER")
    print("="*60)
    
    from surya.settings import settings
    from datetime import datetime
    
    marker_file = Path(settings.MODEL_CACHE_DIR) / ".models_downloaded"
    
    try:
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_content = f"""Models downloaded successfully.
System ready for offline/airgapped operation.
Download completed: {datetime.now().isoformat()}
Surya OCR version: 0.17.0
"""
        marker_file.write_text(marker_content)
        print(f"[OK] Created offline marker: {marker_file}")
        return True
    except Exception as e:
        print(f"[FAILED] Failed to create offline marker: {e}")
        return False

def print_gpu_info():
    """Print GPU information"""
    print("\n" + "="*60)
    print("GPU INFORMATION")
    print("="*60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("CUDA not available - running on CPU")
    except Exception as e:
        print(f"Error getting GPU info: {e}")

def main():
    """Main download process"""
    print("="*60)
    print("SURYA OCR MODEL DOWNLOAD SCRIPT v0.17.0")
    print("="*60)
    print("This will download all required models for offline operation")
    print("Internet access is required for this step only")
    print()
    
    # Print GPU info
    print_gpu_info()
    
    # Setup environment
    setup_environment()
    
    # Download all model types
    download_functions = [
        ("Detection Models", download_detection_models),
        ("Foundation Models", download_foundation_models),
        ("Recognition Models", download_recognition_models),
        ("Layout Models", download_layout_models),
        ("Table Recognition Models", download_table_rec_models),
        ("OCR Error Models", download_ocr_error_models),
    ]
    
    results = {}
    
    for model_type, download_func in download_functions:
        print(f"\n>> Starting: {model_type}")
        try:
            result = download_func()
            results[model_type] = result
            if result:
                print(f">> {model_type}: SUCCESS")
            else:
                print(f">> {model_type}: FAILED")
        except Exception as e:
            print(f">> {model_type}: CRASHED - {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = False
    
    # Test full model loading
    print(f"\n>> Testing complete model loading...")
    load_test_result = test_model_loading()
    results["Model Loading Test"] = load_test_result
    
    # Verify cache
    print(f"\n>> Verifying model cache...")
    cache_result = verify_model_cache()
    results["Cache Verification"] = cache_result
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    failed_downloads = [name for name, result in results.items() if not result]
    
    for name, result in results.items():
        status = "[OK] SUCCESS" if result else "[FAILED]"
        print(f"  {name:30} {status}")
    
    if not failed_downloads:
        print("\n" + "="*60)
        print("ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print("Creating offline marker...")
        create_offline_marker()
        print()
        print("NEXT STEPS:")
        print("1. Stop this container: docker-compose -f docker-compose-download.yml down")
        print("2. Start airgapped container: docker-compose up -d")
        print("3. The container will now run with network restrictions")
        print()
        return 0
    else:
        print(f"\n{len(failed_downloads)} downloads failed:")
        for name in failed_downloads:
            print(f"  - {name}")
        print("\nPlease check your internet connection and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



