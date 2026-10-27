# domain-rag-chatbot
# Domain-Specific RAG Chatbot

A chatbot that answers questions from uploaded PDF documents. It retrieves the most relevant passages and generates answers based only on them, showing the source document and page number.

## Features
- Upload one or more PDFs (type and size validated)
- Text extraction with page metadata
- Chunking with overlap
- Embeddings with Sentence Transformers (all-MiniLM-L6-v2)
- FAISS vector search
- Answers from Groq LLM, grounded in retrieved context
- Source document and page shown under each answer
- Refuses to answer when information is not in the documents

## Workflow
Upload PDFs -> Extract text -> Split into chunks -> Create embeddings -> Store in FAISS -> User asks a question -> Retrieve top chunks -> Send context + question to LLM -> Show answer with sources

## Tech Stack
Python, pypdf, LangChain, Sentence Transformers, FAISS, Groq, Streamlit

## Setup
1. Clone the repo and open the folder.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your Groq key:
   ```
   GROQ_API_KEY=your_key_here
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```

## Usage
1. Upload PDFs in the sidebar.
2. Click **Process Documents**.
3. Ask questions in the chat box.
4. Check the sources shown below each answer.

## Project Structure
```
domain-rag-chatbot/
|-- app.py
|-- rag_pipeline.py
|-- document_loader.py
|-- vector_store.py
|-- prompt.py
|-- requirements.txt
|-- sample_docs/
```

## Safety Notes
- API keys are stored in `.env` and never committed.
- The assistant ignores instructions inside documents.
- Answers may be wrong. Verify high-stakes information in the original documents.
- Do not upload confidential documents without permission.