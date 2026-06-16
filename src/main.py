import json
import logging
from loaders.pdf_loader import load_pdf
from processing.chunking import create_chunks
from embeddings.embedding_manager import EmbeddingManager
from vectordb.chroma_store import ChromaStore
from rag.pipeline import RAGRetriever, RAG_output

# Setup standard logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # logger.info("Initializing talk-to-pdf pipeline...")
    
    # 1. Load PDFs
    pdfs = load_pdf()
    if not pdfs:
        # logger.warning("No PDF documents found to ingest.")
        return

    # 2. Chunk Documents
    chunks = create_chunks(pdfs)
    if not chunks:
        # logger.warning("No chunks created from loaded documents.")
        return

    # 3. Generate embeddings
    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # 4. Ingest into ChromaDB
    vector_store = ChromaStore()
    vector_store.add_documents(chunks, embeddings)

    # 5. Initialize Retriever
    rag_retriever = RAGRetriever(vector_store, embedding_manager)

    # 6. Execute simple query on the RAG pipeline
    print("\n" + "=" * 21 + " QUESTION " + "=" * 21)
    query = input("Your question: ") # e.g. Tell me about Machine Learning
    print("=" * 52 + "\n")
    # logger.info(f"Running query: '{query}'")
    answer = RAG_output(query, rag_retriever, min_score=0.1, return_context=True)
    
    print("\n" + "=" * 20 + " RAG ANSWER " + "=" * 20)
    # print(answer) ## prints normal python dictionary
    print(json.dumps(answer, indent=4)) ## prints in JSON format
    print("=" * 52 + "\n")

if __name__ == "__main__":
    main()