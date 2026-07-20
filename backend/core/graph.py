"""Knowledge graph loading and graph-based retrieval."""

import os
import pickle
import re
from collections import defaultdict

import numpy as np
import networkx as nx

from .config import (
    GRAPH_CACHE_PATH, RRF_CONSTANT, RELATION_SET,
    CONTRADICTORY_RELATIONS, FUZZY_MATCH_THRESHOLD,
)


def canonicalise_relation(rel):
    rel = re.sub(r"[^A-Za-z0-9]+", "_", str(rel).strip().upper()).strip("_")
    aliases = {
        "REQUIRE": "REQUIRES", "REQUIRED_TO": "REQUIRES",
        "PROHIBIT": "PROHIBITS", "PROHIBITED_FROM": "PROHIBITS",
        "ALLOW": "PERMITS", "ALLOWS": "PERMITS", "PERMIT": "PERMITS",
        "GOVERN": "GOVERNS", "APPLIES": "APPLIES_TO", "APPLY_TO": "APPLIES_TO",
        "HAS_OBLIGATIONS": "HAS_OBLIGATION", "SUPERVISE": "SUPERVISES",
        "REPORT_TO": "REPORTS_TO", "MUST_REPORT_TO": "REPORTS_TO",
        "AUTHORISES": "AUTHORIZES", "AUTHORISE": "AUTHORIZES",
        "ENSURE": "ENSURES", "EXEMPT_FROM": "EXEMPTS",
        "DEFINED_AS": "DEFINES", "DEFINE": "DEFINES",
        "DISCLOSE_TO": "DISCLOSES_TO", "NOTIFY": "NOTIFIES",
        "COOPERATE_WITH": "COOPERATES_WITH",
    }
    rel = aliases.get(rel, rel)
    return rel if rel in RELATION_SET else "RELATED_TO"


def normalise_entity(entity, nlp):
    entity = str(entity).lower().strip().replace("_", " ")
    entity = re.sub(r"[^a-z0-9\s-]", "", entity)
    entity = re.sub(r"\s+", " ", entity).strip()
    doc = nlp(entity)
    return " ".join(token.lemma_ for token in doc).strip()


def load_graph(path=GRAPH_CACHE_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Graph cache not found at {path}")


def build_graph_from_corpus(corpus, nlp, extract_fn):
    G = nx.DiGraph()

    def _add_node_source(node, source_id):
        if not G.has_node(node):
            G.add_node(node, source_ids=[])
        ids = G.nodes[node].setdefault("source_ids", [])
        if source_id not in ids:
            ids.append(source_id)

    def _add_edge(subject, relation, obj, source_id):
        if G.has_edge(subject, obj):
            data = G[subject][obj]
            labels = set(data.get("labels", []))
            labels.add(relation)
            data["labels"] = sorted(labels)
            ids = data.setdefault("source_ids", [])
            if source_id not in ids:
                ids.append(source_id)
        else:
            G.add_edge(subject, obj, labels=[relation], source_ids=[source_id])

    for item in corpus:
        elements = extract_fn(item["text"])
        source_id = item["source_id"]

        for entity in elements["entities"]:
            node = normalise_entity(entity, nlp)
            if node:
                _add_node_source(node, source_id)

        for rel in elements["relationships"]:
            if len(rel) != 3:
                continue
            s = normalise_entity(rel[0], nlp)
            relation = canonicalise_relation(rel[1])
            o = normalise_entity(rel[2], nlp)
            if not s or not o:
                continue
            _add_node_source(s, source_id)
            _add_node_source(o, source_id)
            _add_edge(s, relation, o, source_id)

    return G


def save_graph(G, path=GRAPH_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


# --- Graph retrieval functions ---

def embed_graph_nodes(G, embedder):
    node_list = list(G.nodes())
    if not node_list:
        return node_list, np.zeros((0, 384))
    node_embeddings = embedder.encode(
        node_list, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    return node_list, node_embeddings


def find_relevant_nodes(query, node_list, node_embeddings, embedder, top_k):
    query_emb = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]
    scores = node_embeddings @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(node_list[i], float(scores[i])) for i in top_indices]


def _incident_edges(graph, node, exclude_relations=None):
    exclude_relations = set(exclude_relations or [])
    for v in graph.successors(node):
        data = graph[node][v]
        for rel in data.get("labels", ["RELATED_TO"]):
            if rel in exclude_relations:
                continue
            yield node, rel, v, list(data.get("source_ids", []))
    for u in graph.predecessors(node):
        data = graph[u][node]
        for rel in data.get("labels", ["RELATED_TO"]):
            if rel in exclude_relations:
                continue
            yield u, rel, node, list(data.get("source_ids", []))


def get_graph_context(query, G, node_list, node_embeddings, embedder,
                      hops, top_k, top_n, exclude_relations=None):
    seed_nodes = find_relevant_nodes(query, node_list, node_embeddings, embedder, top_k)
    candidates = {}

    for seed, seed_score in seed_nodes:
        frontier = {seed}
        visited = {seed}

        for depth in range(1, hops + 1):
            next_frontier = set()
            for current in frontier:
                for u, rel, v, source_ids in _incident_edges(G, current, exclude_relations):
                    key = (u, rel, v)
                    item = {
                        "subject": u, "relation": rel, "object": v,
                        "depth": depth, "seed": seed,
                        "seed_score": seed_score, "source_ids": source_ids,
                    }
                    prev = candidates.get(key)
                    if prev is None or (depth, -seed_score) < (prev["depth"], -prev["seed_score"]):
                        candidates[key] = item
                    if u not in visited:
                        next_frontier.add(u)
                    if v not in visited:
                        next_frontier.add(v)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

    evidence = list(candidates.values())
    if not evidence:
        return [], seed_nodes

    texts = [f"{x['subject']} {x['relation'].replace('_', ' ')} {x['object']}" for x in evidence]
    triple_embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    triple_scores = triple_embeddings @ query_embedding

    for item, similarity in zip(evidence, triple_scores):
        item["triple_similarity"] = float(similarity)
        item["rank_score"] = 0.65 * float(similarity) + 0.25 * item["seed_score"] + 0.10 / item["depth"]

    evidence.sort(key=lambda x: x["rank_score"], reverse=True)
    return evidence[:top_n], seed_nodes


def format_graph_evidence(items):
    return "\n".join(
        f"[G{i}] {x['subject']} -[{x['relation']}]-> {x['object']} "
        f"(hop={x['depth']}; sources={','.join(x['source_ids'])})"
        for i, x in enumerate(items, 1)
    )


def rank_graph_sources(query, G, node_list, node_embeddings, embedder, corpus_by_id,
                       seed_k, hops, triple_top_n, source_limit=50,
                       exclude_relations=None):
    graph_evidence, seed_nodes = get_graph_context(
        query, G, node_list, node_embeddings, embedder,
        hops=hops, top_k=seed_k, top_n=triple_top_n,
        exclude_relations=exclude_relations,
    )

    scores = defaultdict(float)
    best_triple_rank = {}

    for rank, item in enumerate(graph_evidence, 1):
        valid_sources = [sid for sid in item.get("source_ids", []) if sid in corpus_by_id]
        if not valid_sources:
            continue
        contribution = 1.0 / ((RRF_CONSTANT + rank) * len(valid_sources))
        for sid in valid_sources:
            scores[sid] += contribution
            best_triple_rank[sid] = min(best_triple_rank.get(sid, rank), rank)

    ranked = sorted(scores, key=lambda sid: (-scores[sid], best_triple_rank[sid], sid))[:source_limit]
    return ranked, graph_evidence, seed_nodes, dict(scores)


# --- Symbolic claim checking ---

def fuzzy_match_node(entity_normalised, node_list, node_embeddings, embedder, threshold=FUZZY_MATCH_THRESHOLD):
    if not node_list:
        return None, 0.0
    q = embedder.encode([entity_normalised], convert_to_numpy=True, normalize_embeddings=True)[0]
    similarities = node_embeddings @ q
    idx = int(np.argmax(similarities))
    score = float(similarities[idx])
    if score >= threshold:
        return node_list[idx], score
    return None, score


def resolve_node(entity, G, node_list, node_embeddings, embedder, nlp):
    normalised = normalise_entity(entity, nlp)
    if G.has_node(normalised):
        return normalised, "exact", 1.0
    node, score = fuzzy_match_node(normalised, node_list, node_embeddings, embedder)
    if node:
        return node, "fuzzy", score
    return None, "none", score


def check_claim_symbolically(G, subject, relation, obj, node_list, node_embeddings, embedder, nlp):
    relation = canonicalise_relation(relation)
    s_node, s_type, s_score = resolve_node(subject, G, node_list, node_embeddings, embedder, nlp)
    o_node, o_type, o_score = resolve_node(obj, G, node_list, node_embeddings, embedder, nlp)

    result = {
        "subject": subject, "relation": relation, "object": obj,
        "subject_resolved": s_node, "object_resolved": o_node,
        "subject_match_type": s_type, "object_match_type": o_type,
        "subject_match_score": round(s_score, 3),
        "object_match_score": round(o_score, 3),
        "status": None, "evidence": [], "evidence_source_ids": [], "reason": "",
    }

    if not s_node or not o_node:
        result.update(status="NOT_IN_GRAPH", reason="One or both entities could not be resolved.")
        return result

    labels, source_ids = _direct_edge_labels(G, s_node, o_node)

    if labels:
        contradictions = [l for l in labels if l in CONTRADICTORY_RELATIONS.get(relation, set())]
        if contradictions:
            result.update(
                status="VIOLATION",
                evidence=[f"{s_node} -[{l}]-> {o_node}" for l in contradictions],
                evidence_source_ids=source_ids,
                reason="A direct KG relation conflicts with the claimed relation.",
            )
            return result
        if relation in labels:
            result.update(
                status="SUPPORTED",
                evidence=[f"{s_node} -[{relation}]-> {o_node}"],
                evidence_source_ids=source_ids,
                reason="Exact direct entity + canonical-relation KG match.",
            )
            return result
        result.update(
            status="UNVERIFIED",
            evidence=[f"{s_node} -[{l}]-> {o_node}" for l in labels],
            evidence_source_ids=source_ids,
            reason="Entities share a direct edge, but not with the claimed relation.",
        )
        return result

    candidate = _shortest_path_candidate(G, s_node, o_node)
    if candidate:
        result.update(
            status="UNVERIFIED", evidence=candidate,
            evidence_source_ids=list(dict.fromkeys(
                sid for edge in candidate for sid in edge["source_ids"]
            )),
            reason="A short graph path exists, but connectivity alone is not legal support.",
        )
        return result

    result.update(status="UNSUPPORTED", reason="No supporting relation or short path found.")
    return result


def _direct_edge_labels(graph, subject, obj):
    if not graph.has_edge(subject, obj):
        return [], []
    data = graph[subject][obj]
    return list(data.get("labels", [])), list(data.get("source_ids", []))


def _shortest_path_candidate(graph, subject, obj, max_edges=3):
    try:
        path = nx.shortest_path(graph, subject, obj)
    except nx.NetworkXNoPath:
        return None
    if len(path) - 1 > max_edges:
        return None
    evidence = []
    for u, v in zip(path[:-1], path[1:]):
        labels, source_ids = _direct_edge_labels(graph, u, v)
        evidence.append({"subject": u, "relations": labels, "object": v, "source_ids": source_ids})
    return evidence
