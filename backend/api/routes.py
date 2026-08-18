"""FastAPI routes for the compliance-aware RAG API."""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional

from ..core.pipeline import CompliancePipeline

router = APIRouter()

_app_state = {}


def set_app_state(state):
    _app_state.update(state)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    strategy: str = Field(
        default="legal_hybrid",
        description="Retrieval strategy: dense_vector, legal_vector, graph, legal_hybrid",
    )


class HealthResponse(BaseModel):
    status: str
    corpus_chunks: int
    graph_nodes: int
    graph_edges: int


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        corpus_chunks=len(_app_state.get("corpus", [])),
        graph_nodes=_app_state["G"].number_of_nodes() if "G" in _app_state else 0,
        graph_edges=_app_state["G"].number_of_edges() if "G" in _app_state else 0,
    )


@router.get("/strategies")
def list_strategies():
    return {
        "strategies": [
            {
                "id": "dense_vector",
                "name": "Dense Vector",
                "description": "Standard semantic similarity retrieval using FAISS + OpenAI embeddings.",
            },
            {
                "id": "legal_vector",
                "name": "Legal-aware Vector",
                "description": "Dense vector with Article-heading resolution for explicit regulatory citations.",
            },
            {
                "id": "graph",
                "name": "Graph-only",
                "description": "Knowledge graph BFS expansion with semantic triple reranking.",
            },
            {
                "id": "legal_hybrid",
                "name": "Legal-aware Hybrid (Recommended)",
                "description": "Weighted RRF fusion of legal-aware vector + graph retrieval. Best overall performance.",
            },
        ]
    }


@router.get("/graph/stats")
def graph_stats():
    G = _app_state.get("G")
    if G is None:
        raise HTTPException(status_code=503, detail="Graph not loaded")

    from collections import Counter
    edge_labels = Counter()
    for _, _, data in G.edges(data=True):
        for label in data.get("labels", []):
            edge_labels[label] += 1

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "relation_assertions": sum(edge_labels.values()),
        "relation_distribution": dict(edge_labels.most_common()),
        "sample_nodes": list(G.nodes())[:20],
    }


@router.post("/ask")
def ask_question(
    request: QueryRequest,
    x_openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"),
):
    strategy_map = {
        "dense_vector": "vector",
        "legal_vector": "legal_vector",
        "graph": "graph",
        "legal_hybrid": "legal_hybrid",
    }

    strategy = strategy_map.get(request.strategy)
    if strategy is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{request.strategy}'. Use: {list(strategy_map.keys())}",
        )

    try:
        pipeline = CompliancePipeline(
            api_key=x_openai_api_key,
            corpus=_app_state["corpus"],
            corpus_by_id=_app_state["corpus_by_id"],
            article_index=_app_state["article_index"],
            G=_app_state["G"],
            node_list=_app_state["node_list"],
            node_embeddings=_app_state["node_embeddings"],
            embedder=_app_state["embedder"],
            nlp=_app_state["nlp"],
            vectorstore=_app_state["vectorstore"],
        )
        result = pipeline.run(request.question, strategy=strategy)
        return result
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {error_msg}")
