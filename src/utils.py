import os
import transformers
from typing import Dict, Any, Optional
import warnings

def download_and_cache_models(verbose: bool = True, force_cpu: bool = True, 
                             lightweight: bool = True) -> Dict[str, Any]:
    """
    Downloads and caches all required models for the scraper.
    
    Args:
        verbose: Whether to print progress information.
        force_cpu: Force models to use CPU even if GPU is available.
        lightweight: Use lightweight models when possible.
        
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
    
    # Check for LLMLingua with lightweight option
    try:
        if lightweight:
            # Use a much smaller model for text simplification
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            if verbose:
                print("Initializing lightweight text summarizer...")
            
            # T5-small is ~60MB instead of several GB for LLMLingua
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            summarizer = AutoModelForSeq2SeqLM.from_pretrained("t5-small", device_map="cpu")
            
            # Create a simple wrapper with the same interface as LLMLingua
            class LightweightTextSummarizer:
                def __init__(self, tokenizer, model):
                    self.tokenizer = tokenizer
                    self.model = model
                
                def compress_prompt(self, text, ratio=0.3):
                    """Compress text by summarizing it"""
                    try:
                        prefix = "summarize: "
                        inputs = self.tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512)
                        summary_ids = self.model.generate(
                            inputs["input_ids"], 
                            max_length=150, 
                            min_length=40,
                            length_penalty=2.0, 
                            num_beams=4
                        )
                        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        return {"compressed_prompt": summary}
                    except Exception as e:
                        print(f"Summarization error: {e}")
                        return {"compressed_prompt": text}
                
                # Add the compress method to match the API expected by WebScraper
                def compress(self, text):
                    """Alias for compress_prompt to ensure compatibility"""
                    return self.compress_prompt(text)
            
            models["llm_lingua"] = LightweightTextSummarizer(tokenizer, summarizer)
        else:
            # Use regular LLMLingua if lightweight is False
            from llmlingua import PromptCompressor as LLMLingua  # type: ignore
            if verbose:
                print("Initializing LLMLingua...")
            llm_lingua = LLMLingua(device_map="cpu")  # Force CPU usage
            models["llm_lingua"] = llm_lingua
    except ImportError as e:
        if verbose:
            print(f"Warning: Text simplification module not found: {e}")
            print("Text simplification will be disabled")
    except Exception as e:
        if verbose:
            print(f"Error initializing text simplification: {e}")
    
    if verbose:
        print("All models downloaded and cached successfully!")
        
    return models