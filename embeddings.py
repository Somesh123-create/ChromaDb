from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns a free, open-source embedding model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="cachesemd")
