import streamlit as st

from document_loader import validate_file, load_pdf, chunk_documents
from vector_store import build_vector_store
from rag_pipeline import answer_question
from prompt import FALLBACK_MESSAGE

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="📄")
st.title("📄 Domain-Specific RAG Chatbot")

if "store" not in st.session_state:
    st.session_state.store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uploads:
            st.warning("Please upload at least one PDF.")
        else:
            all_docs, names = [], []
            for f in uploads:
                error = validate_file(f.name, f.size)
                if error:
                    st.error(error)
                    continue
                all_docs.extend(load_pdf(f, f.name))
                names.append(f.name)

            if not all_docs:
                st.error("No readable text found. Scanned PDFs need OCR.")
            else:
                with st.spinner("Processing documents..."):
                    chunks = chunk_documents(all_docs)
                    st.session_state.store = build_vector_store(chunks)
                    st.session_state.files = names
                    st.session_state.messages = []
                st.success(f"Processed {len(names)} file(s), {len(chunks)} chunks.")

    if st.session_state.files:
        st.write("**Loaded files:**")
        for n in st.session_state.files:
            st.write(f"- {n}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Documents"):
        st.session_state.store = None
        st.session_state.files = []
        st.session_state.messages = []
        st.rerun()

st.caption("Always verify high-stakes information in the original documents.")

# ---------- Chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            st.caption("Sources: " + ", ".join(f"{s} (page {p})" for s, p in m["sources"]))

# ---------- Chat input ----------
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if st.session_state.store is None:
            reply, sources = "Please upload and process documents first.", []
        else:
            with st.spinner("Thinking..."):
                reply, sources = answer_question(st.session_state.store, question)
            if FALLBACK_MESSAGE in reply:
                sources = []
        st.write(reply)
        if sources:
            st.caption("Sources: " + ", ".join(f"{s} (page {p})" for s, p in sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )