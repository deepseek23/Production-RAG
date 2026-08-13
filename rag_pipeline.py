from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile
import os
import hashlib

load_dotenv()







embedding_model = HuggingFaceEmbeddings(
        model="BAAI/bge-small-en-v1.5"
    )


KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

def create_kb():
    '''
    create a vector store for knowlegde base
    '''
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,
                   metadata={"source": "Langchain_Knowlegde_base.md"})
    chunks = splitter.split_documents([doc])

    persist_dir = os.path.abspath("./chroma_db")
    hash_path = os.path.join(persist_dir, "docs_hash.txt")

    # compute a hash of the current documents so we can detect changes
    def _compute_hash(docs):
        h = hashlib.sha256()
        for d in docs:
            # include both content and source metadata to be conservative
            content = (d.page_content or "") + (str(d.metadata) if getattr(d, 'metadata', None) else "")
            h.update(content.encode("utf-8"))
        return h.hexdigest()

    current_hash = _compute_hash(chunks)

    # If a persisted DB exists and the stored hash matches, load it instead of rebuilding
    if os.path.isdir(persist_dir) and any(os.scandir(persist_dir)):
        try:
            with open(hash_path, "r", encoding="utf-8") as f:
                saved_hash = f.read().strip()
        except Exception:
            saved_hash = None

        if saved_hash == current_hash:
            # Attempt to load an existing Chroma DB without re-ingesting documents
            try:
                # Try common constructor signatures
                try:
                    vector_store = Chroma(
                        persist_directory=persist_dir,
                        embedding=embedding_model,
                        collection_name='my_chroma_db'
                    )
                except TypeError:
                    vector_store = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=embedding_model,
                        collection_name='my_chroma_db'
                    )
                return vector_store
            except Exception:
                # Fall through to rebuild if loading fails for any reason
                pass

    # create a vector store from the chunks and persist it
    if Chroma is None:
        raise ImportError(
            "Chroma vector store not available. Install 'langchain-chroma' or ensure your LangChain version provides 'langchain.vectorstores.Chroma'."
        )

    os.makedirs(persist_dir, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name='my_chroma_db',
    )

    # write the new hash for future runs
    try:
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(current_hash)
    except Exception:
        pass

    return vector_store


def demo_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = init_chat_model("ollama:llama3.1:8b", temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain
    # Test
    questions = [
        input(str("enter the question: "))
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    demo_rag()