import logging
from loaders.pdf_loader import load_pdf
from processing.chunking import create_chunks
from embeddings.embedding_manager import EmbeddingManager
from vectordb.chroma_store import ChromaStore
from indexing.state_tracker import DocumentStateTracker

logger = logging.getLogger(__name__)

def rebuild_index(tracker: DocumentStateTracker, current_state: dict) -> bool:
    """
    Performs a clean, full rebuild of the vector database index.
    
    Why a full-rebuild indexing strategy is used:
    1. Simplicity and Reliability: Implementing incremental or differential updates requires complex logic 
       to detect file-level content shifts, page deletions, or text alterations. It also requires keeping track 
       of which chunks map to which sections of a file in the vector database to perform partial deletions.
    2. Absolute Consistency: A full rebuild completely resets the database, eliminating the risk of 
       "ghost" or orphaned vectors from deleted files or outdated chunk structures. This guarantees 
       that the database content matches the folder content perfectly.
    3. Low Ingest Cost: For small-to-medium datasets (typical of personal or localized PDF chat applications), 
       re-embedding documents is fast enough that building a complex differential sync engine is not justified.
       
    Why UUID-based document IDs are acceptable under a full-rebuild approach:
    1. Clean-slate execution: Since the existing collection is completely deleted before a rebuild, there are 
       no existing vectors to compare against, replace, or merge with.
    2. Overwrite protection is handled by collection deletion: Under incremental syncing, deterministic IDs 
       (e.g., hash-based or filepath+chunk index) are required to locate and overwrite old documents. Since 
       the collection is empty before chunk addition, we do not need overwrite logic.
    3. Collision-free simplicity: Random UUIDs are simple to generate, stateless, and mathematically 
       guarantee uniqueness across insertions.
    """
    logger.info("Starting clean indexing rebuild flow...")

    # 1. Initialize DB connector and embedding manager
    vector_store = ChromaStore()
    embedding_manager = EmbeddingManager()

    # 2. Reset/delete the collection to prevent duplicate insertions
    vector_store.reset_store()

    # 3. Load PDF documents from configured data directory
    logger.info("Scanning and loading documents...")
    pdfs = load_pdf()
    if not pdfs:
        logger.warning("No PDF documents found in data directory. Index is now empty.")
        tracker.save_state(current_state)
        return False

    # 4. Create text chunks
    logger.info("Generating text chunks from PDFs...")
    chunks = create_chunks(pdfs)
    if not chunks:
        logger.warning("No text chunks could be extracted from loaded PDFs.")
        tracker.save_state(current_state)
        return False

    # 5. Generate embeddings for the chunks
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # 6. Add documents to ChromaDB store
    logger.info("Ingesting chunks and embeddings into ChromaDB...")
    vector_store.add_documents(chunks, embeddings)

    # 7. Update the cached state to avoid future rebuilds
    tracker.save_state(current_state)
    logger.info("Index rebuilding completed successfully.")
    return True
