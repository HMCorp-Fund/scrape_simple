from typing import Dict, Any
import re
import warnings
import numpy as np # type: ignore

# Suppress specific numpy warnings from navec/slovnet
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")

try:
    from natasha import ( # type: ignore
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        Doc
    )
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False

class RussianTextSimplifier:
    def __init__(self):
        if not NATASHA_AVAILABLE:
            raise ImportError("Natasha library is required for Russian text simplification. Install it with: pip install natasha")
        
        # Initialize Natasha components with error handling
        try:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            self.emb = NewsEmbedding()
            self.morph_tagger = NewsMorphTagger(self.emb)
            self.initialized = True
        except Exception as e:
            print(f"Warning: Error initializing Natasha components: {e}")
            self.initialized = False
        
    def simplify_text(self, text, simplification_level=0.3):
        """
        Simplify Russian text while preserving content integrity.
        simplification_level: 0.0 (no simplification) to 1.0 (maximum simplification)
        A more conservative default of 0.3 is used to ensure content integrity.
        """
        if not text or not isinstance(text, str):
            return text
            
        # Safety check for initialization
        if not hasattr(self, 'initialized') or not self.initialized:
            return text
            
        # Safer implementation with error handling
        try:
            # For very short texts, don't simplify
            if len(text.split()) < 10:
                return text
                
            # Create document
            doc = Doc(text)
            
            # Apply with error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                doc.segment(self.segmenter)
                doc.tag_morph(self.morph_tagger)
            
                # For each token, get lemma and POS
                for token in doc.tokens:
                    token.lemmatize(self.morph_vocab)
            
            # Define function words to potentially remove
            function_words_pos = ['CONJ', 'PART', 'PRCL', 'INTJ']
            
            # Build simplified text with fallback mechanisms
            simplified_sentences = []
            for sent in doc.sents:
                # Keep all content words, selectively keep function words
                tokens_to_keep = []
                for token in sent.tokens:
                    # Skip tokens where POS tagging failed
                    if not hasattr(token, 'pos') or token.pos is None:
                        tokens_to_keep.append(token.text)
                        continue
                        
                    # Always keep content words
                    if token.pos not in function_words_pos:
                        tokens_to_keep.append(token.text)
                    # For function words, keep based on simplification level
                    elif token.pos in function_words_pos and simplification_level < 0.5:
                        tokens_to_keep.append(token.text)
                        
                if tokens_to_keep:
                    simplified_sentences.append(" ".join(tokens_to_keep))
                else:
                    # Fallback to original sentence if processing failed
                    simplified_sentences.append(sent.text)
            
            result = " ".join(simplified_sentences).strip()
            
            # Fallback to original text if result is too short (possible processing error)
            if len(result) < len(text) * 0.5:
                print("Warning: Simplified text is too short, using original")
                return text
                
            return result
            
        except Exception as e:
            print(f"Error in Russian text simplification: {e}")
            return text  # Return original text on error