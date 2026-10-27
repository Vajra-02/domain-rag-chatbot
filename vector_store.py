import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = "vectorstore"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def build_vector_store(chunks):
    """Convert chunks to embeddings and store them in FAISS."""
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(INDEX_DIR)
    return store


def load_vector_store():
    """Reload a saved index, or return None if none exists."""
    if not os.path.exists(INDEX_DIR):
        return None
    return FAISS.load_local(
        INDEX_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve(store, question, k=4):
    """Return the top-k most similar chunks."""
    return store.similarity_search(question, k=k)


if __name__ == "__main__":
    from document_loader import load_pdf, chunk_documents

    folder = "sample_docs"
    docs = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            with open(os.path.join(folder, name), "rb") as f:
                docs.extend(load_pdf(f, name))

    chunks = chunk_documents(docs)
    store = build_vector_store(chunks)

    question = "What is a p-value?"
    for r in retrieve(store, question):
        print(f"[{r.metadata['source']} | page {r.metadata['page']}]")
        print(r.page_content[:200], "\n")