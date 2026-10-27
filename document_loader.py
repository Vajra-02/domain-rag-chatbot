import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_MB = 20


def validate_file(filename, size_bytes):
    """Return an error message, or None if the file is OK."""
    if not filename.lower().endswith(".pdf"):
        return f"{filename}: only PDF files are allowed."
    if size_bytes > MAX_FILE_MB * 1024 * 1024:
        return f"{filename}: larger than {MAX_FILE_MB} MB."
    return None


def load_pdf(file, filename):
    """Read every page; keep file name and page number as metadata."""
    reader = PdfReader(file)
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:  # skip empty pages safely
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": filename, "page": page_number},
            )
        )
    return docs


def chunk_documents(docs, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    # Quick test: reads every PDF in sample_docs/
    folder = "sample_docs"
    all_docs = []
    for name in os.listdir(folder):
        if name.lower().endswith(".pdf"):
            with open(os.path.join(folder, name), "rb") as f:
                all_docs.extend(load_pdf(f, name))

    chunks = chunk_documents(all_docs)
    print(f"Pages with text: {len(all_docs)}")
    print(f"Chunks created: {len(chunks)}")
    if chunks:
        print("\nSample chunk:\n", chunks[0].page_content[:300])
        print("\nMetadata:", chunks[0].metadata)