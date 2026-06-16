import logging
import config
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def create_chunks(documents, chunk_size=None, chunk_overlap=None):
    """
    Splits document lists into character text chunks.
    
    Args:
        documents: List of LangChain documents.
        chunk_size: Size of each chunk (optional).
        chunk_overlap: Overlap between chunks (optional).
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP

    # logger.info(f"Splitting {len(documents)} documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["## ", "### ", "\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    # logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks successfully.")

    return chunks
