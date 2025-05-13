import os
import transformers
from typing import Dict, Any, Optional
import warnings

def download_and_cache_models(verbose: bool = True, force_cpu: bool = True) -> Dict[str, Any]:
    """
    Downloads and caches all required models for the scraper.
    
    Args:
        verbose: Whether to print progress information.
        force_cpu: Force models to use CPU even if GPU is available.
        
    Returns:
        Dictionary containing all loaded models and processors.
    """
    models = {}
    
    if verbose:
        print("Downloading and caching required models...")
    
    # Check GPU availability and set device
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
        if verbose:
            print("Forcing CPU usage for models")
    else:
        # Check if we have GPU support
        try:
            import torch # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and verbose:
                print("No GPU detected, using CPU for models")
        except ImportError:
            device = "cpu"
            if verbose:
                print("PyTorch not available, using CPU for models")
    
    # Download image captioning model
    if verbose:
        print("Loading image captioning model...")
    
    try:
        image_processor = transformers.AutoImageProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            use_fast=True
        )
        image_model = transformers.BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            device_map=device
        )
        
        models["image_processor"] = image_processor
        models["image_model"] = image_model
    except Exception as e:
        if verbose:
            print(f"Error loading image model: {e}")
    
    # Check for LLMLingua
    try:
        from llmlingua import PromptCompressor as LLMLingua  # type: ignore
        if verbose:
            print("Initializing LLMLingua...")
        llm_lingua = LLMLingua(device_map="cpu")  # Force CPU usage
        models["llm_lingua"] = llm_lingua
    except ImportError:
        try:
            from llmlingua import LLMLingua  # type: ignore
            if verbose:
                print("Initializing LLMLingua...")
            llm_lingua = LLMLingua(device_map="cpu")  # Force CPU usage
            models["llm_lingua"] = llm_lingua
        except ImportError:
            if verbose:
                print("Warning: LLMLingua package not found or incompatible. Text simplification will be disabled.")
    except Exception as e:
        if verbose:
            print(f"Error initializing LLMLingua: {e}")
    
    if verbose:
        print("All models downloaded and cached successfully!")
        
    return models