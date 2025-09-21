"""
Embedding and vector store wrapper for semantic similarity.
Supports multiple embedding backends with fallback mechanisms.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import pickle

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import Grok client for embeddings
try:
    from .grok_client import grok_embeddings
    GROK_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GROK_EMBEDDINGS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings for semantic similarity calculations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "data/embeddings_cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.embedding_cache = {}
        
        # Initialize embedding model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        # Try sentence transformers first
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentence transformer model: {self.model_name}")
                # Use HF token from environment if present. Do NOT hard-code tokens here.
                import os
                hf_token = os.getenv('HF_TOKEN')
                if hf_token:
                    os.environ['HF_TOKEN'] = hf_token
                    self.model = SentenceTransformer(self.model_name, use_auth_token=True)
                else:
                    # If no token provided, try to load without auth and let the
                    # underlying library raise a helpful error if the model requires auth.
                    self.model = SentenceTransformer(self.model_name)
                logger.info("Sentence transformer model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {str(e)}")
        
        # Try Grok embeddings if available
        if GROK_EMBEDDINGS_AVAILABLE:
            logger.info("Using Grok embeddings API")
            return
        
        # Fallback to simple TF-IDF
        logger.warning("No embedding model available, using TF-IDF fallback")
        self.model = None
    
    def get_embedding(self, text: str, use_cache: bool = True) -> Optional[List[float]]:
        """
        Get embedding for a text string.
        
        Args:
            text: Input text
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            return None
        
        # Check cache first
        if use_cache:
            text_hash = hash(text.strip())
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        
        try:
            # Try sentence transformers
            if self.model and hasattr(self.model, 'encode'):
                embedding = self.model.encode([text])[0].tolist()
                if use_cache:
                    self.embedding_cache[text_hash] = embedding
                return embedding
            
            # Try Grok embeddings
            elif GROK_EMBEDDINGS_AVAILABLE:
                response = grok_embeddings([text])
                if response.get("ok"):
                    embedding = response.get("embeddings", [])
                    if embedding and len(embedding) > 0:
                        if use_cache:
                            self.embedding_cache[text_hash] = embedding[0]
                        return embedding[0]
            
            # Fallback to TF-IDF
            return self._get_tfidf_embedding(text)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return None
    
    def _get_tfidf_embedding(self, text: str) -> List[float]:
        """Fallback TF-IDF embedding."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            
            # Use a simple TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Fit on the text itself (not ideal but works as fallback)
            tfidf_matrix = vectorizer.fit_transform([text])
            # Convert sparse matrix to a dense 1-D numpy array in a way that static analyzers recognize
            embedding = np.asarray(tfidf_matrix.todense()).ravel()
            
            # Normalize the vector
            embedding = normalize([embedding])[0]
            return embedding.tolist()
            
        except ImportError:
            logger.warning("scikit-learn not available for TF-IDF fallback")
            # Return a simple bag-of-words vector
            return self._get_simple_bow_embedding(text)
        except Exception as e:
            logger.warning(f"TF-IDF embedding failed: {str(e)}")
            return self._get_simple_bow_embedding(text)
    
    def _get_simple_bow_embedding(self, text: str) -> List[float]:
        """Simple bag-of-words embedding as final fallback."""
        import re
        from collections import Counter
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Create word frequency vector
        word_counts = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return [0.0] * 100  # Return zero vector
        
        # Create normalized frequency vector
        max_features = 100
        embedding = [0.0] * max_features
        
        for i, (word, count) in enumerate(word_counts.most_common(max_features)):
            embedding[i] = count / total_words
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            embedding = self.get_embedding(text, use_cache=use_cache)
            embeddings.append(embedding)
        
        return embeddings
    
    def save_cache(self, cache_file: str = "embedding_cache.pkl"):
        """Save embedding cache to disk."""
        try:
            cache_path = self.cache_dir / cache_file
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Embedding cache saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def load_cache(self, cache_file: str = "embedding_cache.pkl"):
        """Load embedding cache from disk."""
        try:
            cache_path = self.cache_dir / cache_file
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Embedding cache loaded from {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {str(e)}")
            self.embedding_cache = {}
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        info = {
            "model_name": self.model_name,
            "model_type": "unknown",
            "cache_size": len(self.embedding_cache),
            "available_backends": []
        }
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            info["available_backends"].append("sentence_transformers")
        
        if GROK_EMBEDDINGS_AVAILABLE:
            info["available_backends"].append("grok_api")
        
        if self.model and hasattr(self.model, 'encode'):
            info["model_type"] = "sentence_transformer"
        elif GROK_EMBEDDINGS_AVAILABLE:
            info["model_type"] = "grok_api"
        else:
            info["model_type"] = "tfidf_fallback"
        
        return info

# Convenience functions
def get_text_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
    """Convenience function to get embedding for a single text."""
    manager = EmbeddingManager(model_name=model_name)
    return manager.get_embedding(text)

def calculate_similarity(text1: str, text2: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    """Calculate cosine similarity between two texts."""
    manager = EmbeddingManager(model_name=model_name)
    
    emb1 = manager.get_embedding(text1)
    emb2 = manager.get_embedding(text2)
    
    if not emb1 or not emb2:
        return 0.0
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(emb1, emb2))
    magnitude1 = sum(a * a for a in emb1) ** 0.5
    magnitude2 = sum(b * b for b in emb2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

# Example usage
if __name__ == "__main__":
    # Test embedding functionality
    manager = EmbeddingManager()
    
    # Test texts
    text1 = "Python programming and machine learning"
    text2 = "Data science with Python and AI"
    text3 = "Cooking and gardening"
    
    # Get embeddings
    emb1 = manager.get_embedding(text1)
    emb2 = manager.get_embedding(text2)
    emb3 = manager.get_embedding(text3)
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"Text 3: {text3}")
    
    if emb1 and emb2 and emb3:
        sim_1_2 = calculate_similarity(text1, text2)
        sim_1_3 = calculate_similarity(text1, text3)
        sim_2_3 = calculate_similarity(text2, text3)
        
        print(f"\nSimilarity 1-2: {sim_1_2:.3f}")
        print(f"Similarity 1-3: {sim_1_3:.3f}")
        print(f"Similarity 2-3: {sim_2_3:.3f}")
    
    # Print model info
    info = manager.get_model_info()
    print(f"\nModel Info: {json.dumps(info, indent=2)}")
