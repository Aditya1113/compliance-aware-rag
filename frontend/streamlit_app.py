"""Streamlit frontend for Compliance-Aware RAG — standalone mode for HF Spaces deployment."""

import streamlit as st
import os
import sys
import time

# --- Page config ---
st.set_page_config(
    page_title="Compliance-Aware RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Add backend to path ---
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)


# --- Load resources (cached so it only runs once) ---
@st.cache_resource(show_spinner="Loading regulatory corpus, knowledge graph, and embeddings...")
def load_pipeline_resources():
    import spacy
    from sentence_transformers import SentenceTransformer
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS

    from backend.core.config import (
        CORPUS_PATH, GRAPH_CACHE_PATH, FAISS_CACHE_DIR,
        LOCAL_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    )
    from backend.core.corpus import load_corpus, build_article_index
    from backend.core.graph import load_graph, embed_graph_nodes

    nlp = spacy.load("en_core_web_sm")
    embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    corpus, corpus_by_id = load_corpus(CORPUS_PATH)
    article_index, _ = build_article_index(corpus)
    G = load_graph(GRAPH_CACHE_PATH)
    node_list, node_embeddings = embed_graph_nodes(G, embedder)

    # Load FAISS — needs a placeholder key for deserialization only
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
        "nlp": nlp,
        "vectorstore": vectorstore,
    }


resources = load_pipeline_resources()

# --- Sidebar ---
st.sidebar.title("Compliance-Aware RAG")
st.sidebar.markdown("**Financial Regulatory QA** with knowledge-graph-augmented retrieval and post-generation validation.")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Your key is sent per-request and never stored. Required for GPT-4o-mini calls.",
)

strategy = st.sidebar.selectbox(
    "Retrieval Strategy",
    options=["legal_hybrid", "legal_vector", "dense_vector", "graph"],
    format_func=lambda x: {
        "legal_hybrid": "Legal-aware Hybrid (Recommended)",
        "legal_vector": "Legal-aware Vector",
        "dense_vector": "Dense Vector",
        "graph": "Graph-only",
    }[x],
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. Your question is decomposed into requirements
2. Evidence is retrieved using the selected strategy
3. An answer is drafted from regulatory evidence
4. Claims are validated against source text + knowledge graph
5. If validation fails, one bounded revision is attempted
6. Final answer is accepted or rejected with full audit trail
""")

st.sidebar.markdown("---")

G = resources["G"]
st.sidebar.markdown(f"""
**Corpus:** MiFID II, MiFIR, Delegated Reg. 2017/565
{len(resources['corpus'])} chunks | {G.number_of_nodes()} graph nodes | {G.number_of_edges()} edges
""")

# --- Main area ---
st.title("Compliance-Aware RAG")
st.markdown("*Financial Regulatory Question Answering with Retrieval Validation*")

with st.expander("Example questions", expanded=False):
    examples = [
        "Under MiFID II Article 16(5), what are the principal responsibilities of an investment firm's independent compliance function?",
        "Under MiFIR Article 26(1), when must an investment firm report a transaction, and to whom?",
        "What information about a client's financial situation and investment objectives is relevant to a suitability assessment under MiFID II?",
        "What are the two purposes of commodity-derivative position limits under MiFID II Article 57?",
        "Under Delegated Regulation 2017/565 Article 21, what organisational requirements apply to an investment firm's risk management function?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}"):
            st.session_state["question_input"] = ex

question = st.text_area(
    "Ask a regulatory compliance question:",
    value=st.session_state.get("question_input", ""),
    height=100,
    placeholder="e.g., Under MiFID II Article 16(5), what are the responsibilities of the compliance function?",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Ask", type="primary", use_container_width=True)

if submit:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    if not question or len(question.strip()) < 10:
        st.error("Please enter a question (at least 10 characters).")
        st.stop()

    strategy_map = {
        "dense_vector": "vector",
        "legal_vector": "legal_vector",
        "graph": "graph",
        "legal_hybrid": "legal_hybrid",
    }

    with st.spinner("Running compliance-aware pipeline... (this takes 30-60s)"):
        start_time = time.time()
        try:
            from backend.core.pipeline import CompliancePipeline

            pipeline = CompliancePipeline(
                api_key=api_key,
                **resources,
            )
            result = pipeline.run(question, strategy=strategy_map[strategy])
            elapsed = time.time() - start_time

        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower() or "invalid" in error_msg.lower():
                st.error("Invalid OpenAI API key. Please check your key in the sidebar.")
            else:
                st.error(f"Pipeline error: {error_msg}")
            st.stop()

    # --- Display results ---

    status = result["final_status"]
    if status == "ACCEPTED":
        st.success(f"ACCEPTED  |  {elapsed:.1f}s  |  Strategy: {result['strategy']}")
    else:
        st.warning(f"REJECTED  |  {elapsed:.1f}s  |  Strategy: {result['strategy']}")

    st.markdown("### Answer")
    st.markdown(result["answer"])

    st.markdown("### Validation")
    val = result["validation"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Support Rate", f"{val['support_rate']:.0%}" if val["support_rate"] is not None else "N/A")
    col_b.metric("Claims", f"{val['supported_claims']}/{val['total_claims']}")
    col_c.metric("Contradictions", val["source_contradictions"])
    col_d.metric("Whole-Answer", val["whole_answer_status"])

    if val.get("requirement_results"):
        st.markdown("#### Question Requirements")
        for req in val["requirement_results"]:
            if req["status"] == "SATISFIED":
                st.markdown(f"- :white_check_mark: **{req['requirement']}** — {req['reason']}")
            elif req["status"] == "MISSING":
                st.markdown(f"- :x: **{req['requirement']}** — {req['reason']}")
            else:
                st.markdown(f"- :warning: **{req['requirement']}** — {req['reason']}")

    if val.get("claims"):
        st.markdown("#### Claim Verification")
        for claim in val["claims"]:
            status_icon = {
                "SUPPORTED_SOURCE": ":white_check_mark:",
                "SUPPORTED_GRAPH": ":large_blue_circle:",
                "CONTRADICTED_SOURCE": ":red_circle:",
                "GRAPH_CONFLICT": ":red_circle:",
                "UNVERIFIED": ":yellow_circle:",
            }.get(claim["final_status"], ":question:")
            st.markdown(
                f"{status_icon} `{claim['subject']}` —[{claim['relation']}]→ `{claim['object']}` : **{claim['final_status']}**"
            )

    if result.get("revised") and result.get("initial_validation"):
        st.markdown("#### Revision")
        iv = result["initial_validation"]
        st.info(
            f"Initial draft was **{'valid' if iv['valid'] else 'invalid'}** "
            f"(support: {iv['support_rate']}, whole-answer: {iv['whole_answer_status']}). "
            f"One revision was attempted."
        )

    with st.expander("Retrieved Document Evidence", expanded=False):
        for doc in result.get("doc_evidence", []):
            st.markdown(f"**{doc['source_id']}** ({doc['doc_id']})")
            st.text(doc["text"])
            st.markdown("---")

    with st.expander("Graph Evidence", expanded=False):
        for triple in result.get("graph_evidence", []):
            st.markdown(
                f"- `{triple['subject']}` —[{triple['relation']}]→ `{triple['object']}` "
                f"(hop={triple['depth']}, sources={', '.join(triple['source_ids'])})"
            )

    with st.expander("Full Pipeline Output (JSON)", expanded=False):
        st.json(result)
