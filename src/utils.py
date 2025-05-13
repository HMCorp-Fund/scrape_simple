import os
import transformers
from typing import Dict, Any, Optional

def download_and_cache_models(verbose: bool = True) -> Dict[str, Any]:
    """
    Downloads and caches all required models for the scraper.
    
    Args:
        verbose: Whether to print progress information.
        
    Returns:
        Dictionary containing all loaded models and processors.
    """
    models = {}
    
    if verbose:
        print("Downloading and caching required models...")
    
    # Download image captioning model
    if verbose:
        print("Loading image captioning model...")
    
    image_processor = transformers.AutoImageProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        use_fast=True
    )
    image_model = transformers.BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    
    models["image_processor"] = image_processor
    models["image_model"] = image_model
    
    # Check for LLMLingua
    try:
        from llmlingua import PromptCompressor as LLMLingua  # type: ignore
        if verbose:
            print("Initializing LLMLingua...")
        llm_lingua = LLMLingua()
        models["llm_lingua"] = llm_lingua
    except ImportError:
        try:
            from llmlingua import LLMLingua  # type: ignore
            if verbose:
                print("Initializing LLMLingua...")
            llm_lingua = LLMLingua()
            models["llm_lingua"] = llm_lingua
        except ImportError:
            if verbose:
                print("Warning: LLMLingua package not found or incompatible. Text simplification will be disabled.")
    
    if verbose:
        print("All models downloaded and cached successfully!")
        
    return models