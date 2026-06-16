import logging
import os
import uuid
from typing import List, Any
import numpy as np
import chromadb
import config

logger = logging.getLogger(__name__)

class ChromaStore:
    """Stores embedded documents inside ChromaDB"""
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Args:
            collection_name: Name of the collection (optional).
            persist_directory: Directory to persist data (optional).
        """
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or str(config.VECTOR_DB_DIR)
        self._client = None
        self._collection = None

    def initialize_store(self):
        """Lazy initializes the ChromaDB client and collection."""
        if self._client is None:
            # logger.info("Initializing ChromaDB client...")
            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                self._client = chromadb.PersistentClient(self.persist_directory)
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Web documents embedded for RAG"}
                )
                # logger.info("ChromaDB client and collection initialized successfully.")
                # logger.info(f"Collection: {self.collection_name}, Count: {self._collection.count()}")
            except Exception as e:
                # logger.error(f"Error initializing Chroma store: {e}")
                raise

    @property
    def client(self):
        self.initialize_store()
        return self._client

    @property
    def collection(self):
        self.initialize_store()
        return self._collection

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the store.
        
        Args:
            documents: List of documents.
            embeddings: List/numpy array of embeddings for the documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must be the same")

        # Accessing collection triggers initialization
        collection = self.collection

        # logger.info(f"Adding {len(documents)} documents to ChromaDB store: {self.collection_name} ...")

        ids = []
        metadatas = []
        document_text = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        try:
            collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=document_text,
            )
            # logger.info(f"{len(documents)} documents added successfully. Collection count: {collection.count()}")
        except Exception as e:
            # logger.error(f"Error adding documents: {e}")
            raise
