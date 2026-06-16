import logging
import config
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

logger = logging.getLogger(__name__)

def load_pdf(data_dir=None):
    """
    Loads all PDF documents from the specified directory.
    
    Args:
        data_dir: Path to directory containing PDFs (optional, defaults to config.DATA_DIR).
    """
    if data_dir is None:
        data_dir = str(config.DATA_DIR)

    # logger.info(f"Loading PDFs from directory: {data_dir} ...")
    
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    documents = pdf_loader.load()
    # logger.info(f"Successfully loaded {len(documents)} document pages/sections.")
    
    return documents
