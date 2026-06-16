import logging
from indexing.state_tracker import DocumentStateTracker
from indexing.build_index import rebuild_index
from embeddings.embedding_manager import EmbeddingManager
from vectordb.chroma_store import ChromaStore
from rag.pipeline import RAGRetriever
from chat.cli import chat_loop

# Setup standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing RAG CLI application...")

    # 1. Initialize the document state tracker
    tracker = DocumentStateTracker()
    current_state = tracker.get_current_state()
    cached_state = tracker.load_cached_state()

    # 2. Check if the index is stale (files added, removed, or modified)
    if tracker.is_index_stale(current_state, cached_state):
        rebuild_index(tracker, current_state)
    else:
        logger.info("PDF documents are unchanged. Skipping database rebuild.")

    # 3. Establish database & model connections for the active session
    logger.info("Connecting to vector database...")
    embedding_manager = EmbeddingManager()
    vector_store = ChromaStore()
    rag_retriever = RAGRetriever(vector_store, embedding_manager)

    # 4. Start the interactive console chat loop
    chat_loop(rag_retriever)

if __name__ == "__main__":
    main()