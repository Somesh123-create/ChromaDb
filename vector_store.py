import chromadb
from langchain.vectorstores import Chroma
from config.settings import chroma_client, embedding_function


def get_vector_store():
    """
    Initializes ChromaDB as a vector store.
    """
    embedding_function = get_embedding_model()
    return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

def store_documents(chunks):
    """
    Stores document chunks in ChromaDB as vectors.
    """
    vector_store = get_vector_store()

    for i, chunk in enumerate(chunks):
        vector_store.add_texts([chunk], metadatas=[{"chunk_id": i}])

    vector_store.persist()


def store_documents(chunks, pdf_name, collection_name):
    """Stores chunks into ChromaDB with metadata."""
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Prepare the text chunks for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings for the chunks
    embeddings = embedding_function.embed_documents(texts)


    # Add chunks with embeddings to ChromaDB
    for i, (chunk_text, embedding) in enumerate(zip(texts, embeddings)):
        collection.add(
            ids=[f"{pdf_name}_{i}"],  # Unique ID for each chunk
            documents=[chunk_text],    # Text chunk
            embeddings=[embedding],    # Embedding vector for semantic search
            metadatas=[{"pdf_name": pdf_name, "page": chunks[i]["page"]}]  # Metadata (PDF name, page number)
        )

    return len(chunks)  # Returns the number of chunks stored in ChromaDB




def semantic_search(query, collection_name="history_docs", top_k=4):
    """Retrieves top-k relevant chunks from a given collection."""

    # Load existing collection
    collection = chroma_client.get_collection(collection_name)

    # Perform similarity search
    results = collection.query(query_texts=[query], n_results=top_k)

    # Return (chunk_text, pdf_name) for top-k matches
    return [(doc, meta["pdf_name"]) for doc, meta in zip(results["documents"][0], results["metadatas"][0])]



# def semantic_search(query, top_k=4):
#     """Search for the most relevant chunks using embeddings."""
#     vector_store = get_vector_store()
#     results = vector_store.similarity_search(query, k=top_k)
#
#     return [(res.page_content, res.metadata["chunk_id"]) for res in results]
