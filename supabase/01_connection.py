import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")


DATABASE_URL = SUPABASE_URL 

def connect_supabase():
    '''connect to pgvector'''
    if not DATABASE_URL:
        raise ValueError("SUPABASE_DATABASE_URL is not set in environment variables.")

    embeddings = HuggingFaceEmbeddings(
        model="BAAI/bge-small-en-v1.5"
    )
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="freecodecamp",
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    return vector_store


def store_sample_documents():
    """Create sample documents and store them in the PGVector collection."""
    vector_store = connect_supabase()

    sample_docs = [
        Document(
            page_content="LangChain helps build LLM applications with retrieval, agents, and chains.",
            metadata={"source": "sample", "topic": "langchain", "doc_id": "doc-1"},
        ),
        Document(
            page_content="PGVector stores embeddings in PostgreSQL for semantic search and RAG pipelines.",
            metadata={"source": "sample", "topic": "pgvector", "doc_id": "doc-2"},
        ),
        Document(
            page_content="Chunking breaks long text into smaller pieces before generating embeddings.",
            metadata={"source": "sample", "topic": "chunking", "doc_id": "doc-3"},
        ),
    ]

    ids = vector_store.add_documents(sample_docs)
    print(f"Stored {len(sample_docs)} sample documents in PGVector.")
    print(f"Inserted IDs: {ids}")
    return ids


def run_similarity_search(query: str = "What is PGVector used for?", k: int = 2):
    """Run a similarity search on PGVector and print results."""
    vector_store = connect_supabase()
    results = vector_store.similarity_search(query, k=k)

    print("\nSimilarity search is working.")
    print(f"Query: {query}")
    print(f"Top {k} results:")
    for idx, doc in enumerate(results, start=1):
        print(f"{idx}. {doc.page_content}")
        print(f"   metadata: {doc.metadata}")

    return results


def demo_store_and_search():
    """Seed sample docs and verify similarity search end-to-end."""
    print("Connecting to Supabase PGVector...")
    connect_supabase()
    print("Connection successful.")

    store_sample_documents()
    run_similarity_search()
    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    demo_store_and_search()