"""Retrieval strategies: Dense Vector, Legal-aware Vector, Graph-only, Legal-aware Hybrid."""

import re
from collections import defaultdict

from .config import (
    VECTOR_CANDIDATE_K, RRF_CONSTANT, RETRIEVAL_EVAL_KS,
    SELECTED_GRAPH_SEED_K, SELECTED_GRAPH_HOPS,
    SELECTED_GRAPH_TRIPLE_TOP_N, VECTOR_RRF_WEIGHT,
    GRAPH_RRF_WEIGHT, EXCLUDED_GRAPH_RELATIONS,
)


def detect_document_id(query):
    q = query.lower()
    if "delegated regulation" in q or "2017/565" in q:
        return "delegated565"
    if "mifir" in q or "600/2014" in q:
        return "mifir"
    if "mifid ii" in q or "mifid2" in q or "2014/65" in q:
        return "mifid2"
    return None


def extract_article_number(query):
    match = re.search(r"\bArticle\s+(\d+[a-zA-Z]?)", query, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def rank_article_sources(query, article_index):
    doc_id = detect_document_id(query)
    article = extract_article_number(query)
    if not doc_id or not article:
        return []
    return article_index.get((doc_id, article), [])


def rank_vector(query, vectorstore, k=VECTOR_CANDIDATE_K):
    docs = vectorstore.similarity_search(query, k=k)
    return [d.metadata["source_id"] for d in docs]


def legal_aware_vector_rank(query, vector_rank, article_index):
    article_rank = rank_article_sources(query, article_index)
    final_rank = []
    for sid in article_rank + list(vector_rank):
        if sid not in final_rank:
            final_rank.append(sid)
    return final_rank


def weighted_rrf(vector_rank, graph_rank,
                 vector_weight=VECTOR_RRF_WEIGHT,
                 graph_weight=GRAPH_RRF_WEIGHT,
                 rrf_constant=RRF_CONSTANT):
    scores = defaultdict(float)
    for rank, sid in enumerate(vector_rank, 1):
        scores[sid] += vector_weight / (rrf_constant + rank)
    for rank, sid in enumerate(graph_rank, 1):
        scores[sid] += graph_weight / (rrf_constant + rank)
    return sorted(scores, key=lambda sid: (-scores[sid], sid))


def retrieve_all_strategies(query, vectorstore, article_index,
                            G, node_list, node_embeddings, embedder, corpus_by_id):
    from .graph import rank_graph_sources, get_graph_context

    k = max(RETRIEVAL_EVAL_KS)
    vector_rank = rank_vector(query, vectorstore, k=k)
    legal_rank = legal_aware_vector_rank(query, vector_rank, article_index)

    graph_rank, graph_evidence_full, seed_nodes, _ = rank_graph_sources(
        query, G, node_list, node_embeddings, embedder, corpus_by_id,
        seed_k=SELECTED_GRAPH_SEED_K, hops=SELECTED_GRAPH_HOPS,
        triple_top_n=SELECTED_GRAPH_TRIPLE_TOP_N,
        source_limit=k * 3, exclude_relations=EXCLUDED_GRAPH_RELATIONS,
    )

    hybrid_rank = weighted_rrf(legal_rank, graph_rank)

    graph_prompt_evidence, _ = get_graph_context(
        query, G, node_list, node_embeddings, embedder,
        hops=SELECTED_GRAPH_HOPS, top_k=SELECTED_GRAPH_SEED_K,
        top_n=min(SELECTED_GRAPH_TRIPLE_TOP_N, 10),
        exclude_relations=EXCLUDED_GRAPH_RELATIONS,
    )

    return {
        "vector": vector_rank,
        "legal_vector": legal_rank,
        "graph": graph_rank,
        "legal_hybrid": hybrid_rank,
        "graph_evidence": graph_prompt_evidence,
        "seed_nodes": seed_nodes,
    }
