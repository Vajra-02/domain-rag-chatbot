import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from vector_store import retrieve
from prompt import SYSTEM_PROMPT

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def format_context(chunks):
    parts = []
    for c in chunks:
        parts.append(
            f"[Source: {c.metadata['source']}, page {c.metadata['page']}]\n{c.page_content}"
        )
    return "\n\n".join(parts)


def answer_question(store, question, k=4):
    chunks = retrieve(store, question, k=k)
    context = format_context(chunks)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | get_llm()
    response = chain.invoke({"context": context, "question": question})

    sources = sorted({(c.metadata["source"], c.metadata["page"]) for c in chunks})
    return response.content, sources


if __name__ == "__main__":
    from vector_store import load_vector_store

    store = load_vector_store()
    for q in ["What is a p-value?", "Who is the company CEO?"]:
        answer, sources = answer_question(store, q)
        print("Q:", q)
        print("A:", answer)
        print("Sources:", sources, "\n")
