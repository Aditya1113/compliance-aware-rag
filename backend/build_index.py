"""One-time script to build the FAISS index. Requires OPENAI_API_KEY env var."""

import os
import sys

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS

    from core.config import CORPUS_PATH, FAISS_CACHE_DIR, OPENAI_EMBEDDING_MODEL
    from core.corpus import load_corpus

    print("Loading corpus...")
    corpus, _ = load_corpus(CORPUS_PATH)
    print(f"  {len(corpus)} chunks")

    print("Building FAISS index (this calls the OpenAI embeddings API)...")
    openai_embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=api_key)
    documents = [
        Document(
            page_content=item["text"],
            metadata={
                "source_id": item["source_id"],
                "doc_id": item["doc_id"],
                "filename": item["filename"],
                "chunk_id": item["chunk_id"],
            },
        )
        for item in corpus
    ]
    vectorstore = FAISS.from_documents(documents, openai_embeddings)

    os.makedirs(FAISS_CACHE_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_CACHE_DIR)
    print(f"FAISS index saved to {FAISS_CACHE_DIR}")


if __name__ == "__main__":
    main()
