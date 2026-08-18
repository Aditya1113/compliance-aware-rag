"""FastAPI application entry point with startup loading."""

import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Compliance-Aware RAG",
    description="Financial regulatory question answering with knowledge-graph-augmented retrieval and post-generation validation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_resources():
    import spacy
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS

    from .core.config import (
        CORPUS_PATH, GRAPH_CACHE_PATH, FAISS_CACHE_DIR,
        LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    )
    from .core.corpus import load_corpus, build_article_index
    from .core.graph import load_graph, embed_graph_nodes
    from .api.routes import set_app_state

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading sentence transformer...")
    embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

    print("Loading frozen corpus...")
    corpus, corpus_by_id = load_corpus(CORPUS_PATH)
    print(f"  {len(corpus)} chunks loaded")

    print("Building article index...")
    article_index, article_metadata = build_article_index(corpus)
    print(f"  {len(article_index)} articles indexed")

    print("Loading knowledge graph...")
    G = load_graph(GRAPH_CACHE_PATH)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Embedding graph nodes...")
    node_list, node_embeddings = embed_graph_nodes(G, embedder)

    # Build FAISS index — requires a temporary OpenAI key or pre-built index
    print("Loading FAISS vector store...")
    if os.path.exists(os.path.join(FAISS_CACHE_DIR, "index.faiss")):
        # Use a dummy key for loading — actual queries use the user's key
        openai_embeddings = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
        )
        vectorstore = FAISS.load_local(
            FAISS_CACHE_DIR, openai_embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"  FAISS index loaded from {FAISS_CACHE_DIR}")
    else:
        build_key = os.environ.get("OPENAI_API_KEY")
        if not build_key:
            print("  WARNING: No FAISS index found and no OPENAI_API_KEY to build one.")
            print("  Run: python -m app.backend.build_index to create the FAISS index first.")
            vectorstore = None
        else:
            openai_embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=build_key)
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
            print(f"  Built and saved FAISS index to {FAISS_CACHE_DIR}")

    set_app_state({
        "corpus": corpus,
        "corpus_by_id": corpus_by_id,
        "article_index": article_index,
        "G": G,
        "node_list": node_list,
        "node_embeddings": node_embeddings,
        "embedder": embedder,
        "nlp": nlp,
        "vectorstore": vectorstore,
    })

    print("All resources loaded. Server ready.")


from .api.routes import router
app.include_router(router, prefix="/api")
