import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    """
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    pdf_text = []

    for doc in documents:
        pdf_text.append({"text": doc.page_content, "page": doc.metadata.get("page", 0)})
    return pdf_text


def chunk_text(pdf_text, chunk_size=500, chunk_overlap=100):
    """
    Splits the extracted text into manageable chunks for embedding.
    Uses RecursiveCharacterTextSplitter with optimized parameters.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Adjust chunk size
        chunk_overlap=chunk_overlap,  # Slight overlap for context retention
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for item in pdf_text:
        split_chunks = text_splitter.split_text(item["text"])
        for chunk in split_chunks:
            chunks.append({"text": chunk, "page": item["page"]})

    return chunks  # Returns list of {"text": ..., "page": ...}
