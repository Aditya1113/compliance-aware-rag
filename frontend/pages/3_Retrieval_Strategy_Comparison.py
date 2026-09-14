"""Side-by-side comparison of all 4 retrieval strategies on the same query."""

import streamlit as st
import os
import sys
import time

st.set_page_config(
    page_title="Retrieval Strategy Comparison",
    page_icon="",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, APP_DIR)


@st.cache_resource(show_spinner="Loading retrieval resources...")
def load_resources():
    import spacy
    from sentence_transformers import SentenceTransformer
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    from backend.core.config import (
        CORPUS_PATH, GRAPH_CACHE_PATH, FAISS_CACHE_DIR,
        LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    )
    from backend.core.corpus import load_corpus, build_article_index
    from backend.core.graph import load_graph, embed_graph_nodes

    embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    corpus, corpus_by_id = load_corpus(CORPUS_PATH)
    article_index, _ = build_article_index(corpus)
    G = load_graph(GRAPH_CACHE_PATH)
    node_list, node_embeddings = embed_graph_nodes(G, embedder)

    openai_embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    )
    vectorstore = FAISS.load_local(
        FAISS_CACHE_DIR, openai_embeddings,
        allow_dangerous_deserialization=True,
    )

    return {
        "corpus": corpus,
        "corpus_by_id": corpus_by_id,
        "article_index": article_index,
        "G": G,
        "node_list": node_list,
        "node_embeddings": node_embeddings,
        "embedder": embedder,
        "vectorstore": vectorstore,
    }


resources = load_resources()

# --- Sidebar ---
st.sidebar.title("Retrieval Comparison")
st.sidebar.markdown("Compare all 4 retrieval strategies on the same query to see how they differ.")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Required for dense vector similarity search.",
)

top_k = st.sidebar.slider("Documents to compare (top-K)", 3, 15, 5)

# --- Main ---
st.title("Retrieval Strategy Comparison")
st.markdown("*See how Dense Vector, Legal-aware Vector, Graph-only, and Legal-aware Hybrid rank documents differently for the same question.*")

query = st.text_area(
    "Enter a regulatory question:",
    height=80,
    placeholder="e.g., Under MiFID II Article 16(5), what are the responsibilities of the compliance function?",
)

submit = st.button("Compare Strategies", type="primary")

if submit:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not query or len(query.strip()) < 10:
        st.error("Please enter a question (at least 10 characters).")
        st.stop()

    from langchain_openai import OpenAIEmbeddings
    from backend.core.config import OPENAI_EMBEDDING_MODEL, GENERATION_DOC_K
    from backend.core.retrieval import retrieve_all_strategies
    from backend.core.corpus import docs_from_source_ids

    # Swap embedding function to user's key
    resources["vectorstore"].embedding_function = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL, api_key=api_key,
    )

    with st.spinner("Running all 4 retrieval strategies..."):
        start = time.time()
        try:
            rankings = retrieve_all_strategies(
                query,
                resources["vectorstore"],
                resources["article_index"],
                resources["G"],
                resources["node_list"],
                resources["node_embeddings"],
                resources["embedder"],
                resources["corpus_by_id"],
            )
            elapsed = time.time() - start
        except Exception as e:
            error_msg = str(e)
            if "Incorrect API key" in error_msg:
                st.error("Invalid OpenAI API key. Please check your key.")
            else:
                st.error(f"Retrieval error: {error_msg}")
            st.stop()

    st.success(f"All strategies completed in {elapsed:.1f}s")

    strategies = {
        "Dense Vector": "vector",
        "Legal-aware Vector": "legal_vector",
        "Graph-only": "graph",
        "Legal-aware Hybrid": "legal_hybrid",
    }

    # Collect top-K for each strategy
    strategy_docs = {}
    all_doc_ids = set()
    for label, key in strategies.items():
        ids = rankings.get(key, [])[:top_k]
        strategy_docs[label] = ids
        all_doc_ids.update(ids)

    # --- Overlap Analysis ---
    st.markdown("---")
    st.markdown("### Overlap Analysis")

    import pandas as pd

    strategy_names = list(strategies.keys())
    overlap_data = []
    for i, s1 in enumerate(strategy_names):
        row = []
        set1 = set(strategy_docs[s1])
        for j, s2 in enumerate(strategy_names):
            set2 = set(strategy_docs[s2])
            if set1 or set2:
                overlap = len(set1 & set2)
                row.append(overlap)
            else:
                row.append(0)
        overlap_data.append(row)

    overlap_df = pd.DataFrame(
        overlap_data,
        index=strategy_names,
        columns=strategy_names,
    )
    st.dataframe(overlap_df, use_container_width=True)
    st.caption(f"Number of shared documents in top-{top_k} results between each pair of strategies.")

    # --- Unique contributions ---
    st.markdown("### Unique Documents per Strategy")
    for label in strategy_names:
        others = set()
        for other_label in strategy_names:
            if other_label != label:
                others.update(strategy_docs[other_label])
        unique = [d for d in strategy_docs[label] if d not in others]
        if unique:
            st.markdown(f"**{label}** brings {len(unique)} unique doc(s): `{'`, `'.join(unique)}`")
        else:
            st.markdown(f"**{label}** — no unique documents (all shared with other strategies)")

    # --- Side-by-side rankings ---
    st.markdown("---")
    st.markdown("### Side-by-Side Rankings")

    cols = st.columns(4)
    for col, (label, key) in zip(cols, strategies.items()):
        with col:
            st.markdown(f"**{label}**")
            ids = strategy_docs[label]
            if not ids:
                st.markdown("*No results*")
                continue
            for rank, sid in enumerate(ids, 1):
                # Check if this doc appears in other strategies
                in_others = sum(1 for other in strategies if other != label and sid in strategy_docs[other])
                shared_badge = f" ({in_others+1}/4)" if in_others > 0 else " (unique)"
                st.markdown(f"{rank}. `{sid}`{shared_badge}")

    # --- Document previews ---
    st.markdown("---")
    st.markdown("### Document Previews")
    st.markdown("Click to expand any retrieved document chunk.")

    all_docs = docs_from_source_ids(list(all_doc_ids), resources["corpus_by_id"])
    doc_map = {d["source_id"]: d for d in all_docs}

    for sid in sorted(all_doc_ids):
        doc = doc_map.get(sid)
        if not doc:
            continue
        # Which strategies retrieved this?
        found_in = [label for label in strategy_names if sid in strategy_docs[label]]
        ranks = []
        for label in found_in:
            r = strategy_docs[label].index(sid) + 1
            ranks.append(f"{label} #{r}")

        with st.expander(f"{sid} — retrieved by: {', '.join(ranks)}", expanded=False):
            st.text(doc["text"][:500] + ("..." if len(doc["text"]) > 500 else ""))

    # --- Graph evidence ---
    graph_evidence = rankings.get("graph_evidence", [])
    if graph_evidence:
        st.markdown("---")
        st.markdown("### Graph Evidence (triples used by Graph & Hybrid strategies)")
        for triple in graph_evidence[:15]:
            st.markdown(
                f"- `{triple['subject']}` --[{triple['relation']}]--> `{triple['object']}` "
                f"(hop={triple['depth']}, sources={', '.join(triple['source_ids'][:3])})"
            )
        if len(graph_evidence) > 15:
            st.caption(f"Showing 15 of {len(graph_evidence)} triples.")
