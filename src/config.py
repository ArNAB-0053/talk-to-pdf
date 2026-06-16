from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

# OPEN ROUTER config
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
OPEN_ROUTER_MODEL_NAME = os.getenv("OPEN_ROUTER_MODEL_NAME")
OPEN_ROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Embedding settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Vector DB settings
CHROMA_COLLECTION_NAME = "pdf_documents"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
RETRIEVAL_TOP_K = 3
