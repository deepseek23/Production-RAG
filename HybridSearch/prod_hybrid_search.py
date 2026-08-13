import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


DB_PATH = os.path.abspath("./chroma_db")
COLLECTION_NAME = "my_chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


# --------------------------------------------------
# Existing documents
# --------------------------------------------------

documents = [
    Document(
        page_content="Python is a high-level programming language used for AI.",
        metadata={"id": "doc1", "category": "programming"}
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications.",
        metadata={"id": "doc2", "category": "AI"}
    ),
    Document(
        page_content="Vector databases store embeddings for semantic search.",
        metadata={"id": "doc3", "category": "database"}
    ),
]


# --------------------------------------------------
# Create / load Chroma
# --------------------------------------------------

if os.path.exists(DB_PATH):

    print("Loading existing Chroma DB...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

else:

    print("Creating Chroma DB...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
    )

    print("Chroma DB created!")


# --------------------------------------------------
# Add NEW documents
# --------------------------------------------------

new_documents = [
    Document(
        page_content=(
            "Hybrid search combines keyword search and "
            "vector search to improve retrieval."
        ),
        metadata={
            "id": "doc4",
            "category": "search"
        }
    ),
    Document(
        page_content=(
            "BM25 is a keyword-based ranking algorithm "
            "used for information retrieval."
        ),
        metadata={
            "id": "doc5",
            "category": "search"
        }
    ),
]


# Add only the new documents
if new_documents:

    print(f"Adding {len(new_documents)} new documents...")

    vectorstore.add_documents(
        documents=new_documents,
        ids=[
            doc.metadata["id"]
            for doc in new_documents
        ]
    )

    print("New documents added!")


# --------------------------------------------------
# Vector retriever
# --------------------------------------------------

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# BM25
# --------------------------------------------------

# IMPORTANT:
# BM25 needs ALL documents that should be searchable.
#
# In this example:
all_documents = documents + new_documents

bm25_retriever = BM25Retriever.from_documents(
    all_documents,
    k=3
)
print("both retievers are ready!")


ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
    rrf_k=50
)
print('hybrid Retiever is ready')

def test_query(query, name, retiever):
    '''test a query and show results'''
    result = retiever.invoke(query)
    print(f'\\n{name}, - Query: \"{query}\"')
    for i, doc in enumerate(result[:3]):
        preview = doc.page_content[:80] + '...'
        print(f'  {i+1}, {preview}')
    return result

# test_queries = [
#     "BM25",
#     "vector databases",
#     "How can keyword and semantic search be combined?",
#     "What framework is used to build applications with large language models?",
#     "How can databases find documents with similar meaning?",
#     "How does RAG retrieve information for a language model?",
# ]

test_queries = [
    'What language use for AI development ?'
]
if __name__ == "__main__":
    for q in test_queries:
        test_query(q, "Hybrid Retrieval", ensemble_retriever)