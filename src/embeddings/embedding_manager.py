import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import config

# logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: Hugging Face model for sentence embeddings (optional).
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model = None

    def _load_model(self):
        if self.model is None:
            # logger.info(f"Loading embedding model: {self.model_name} ...")
            try:
                self.model = SentenceTransformer(self.model_name)
                # logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}")
            except Exception as e:
                # logger.error(f"Error loading model {self.model_name}: {e}")
                raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embedding for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            Numpy array of embeddings.
        """
        if not texts:
            return np.array([])

        # Lazy load the model weights
        self._load_model()

        # logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        # logger.info(f"Embeddings generated successfully. Shape: {embeddings.shape}")
        return embeddings
