import os
from dotenv import load_dotenv
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()  # Load environment variables


# ChromaDB Configuration
CHROMA_DB_PATH = "database/chroma_db"


chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="cachesemd")
