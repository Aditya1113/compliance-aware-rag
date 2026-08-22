"""Streamlit frontend for Compliance-Aware RAG."""

import streamlit as st
import requests
import json
import time

# --- Page config ---
st.set_page_config(
    page_title="Compliance-Aware RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
_default_api = os.environ.get("BACKEND_URL", "http://localhost:8000")
API_URL = st.sidebar.text_input("Backend URL", value=_default_api, help="FastAPI backend URL")

# --- Sidebar ---
st.sidebar.title("Compliance-Aware RAG")
st.sidebar.markdown("**Financial Regulatory QA** with knowledge-graph-augmented retrieval and post-generation validation.")
st.sidebar.markdown("---")

# API Key input
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Your key is sent per-request and never stored on the server.",
)

# Strategy selector
strategy = st.sidebar.selectbox(
    "Retrieval Strategy",
    options=["legal_hybrid", "legal_vector", "dense_vector", "graph"],
    format_func=lambda x: {
        "legal_hybrid": "Legal-aware Hybrid (Recommended)",
        "legal_vector": "Legal-aware Vector",
        "dense_vector": "Dense Vector",
        "graph": "Graph-only",
    }[x],
    help="Choose which retrieval strategy powers the answer.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. Your question is decomposed into requirements
2. Evidence is retrieved using the selected strategy
3. An answer is drafted from regulatory evidence
4. Claims are extracted and validated against source text + knowledge graph
5. If validation fails, one bounded revision is attempted
6. Final answer is accepted or rejected with full audit trail
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Corpus:** MiFID II, MiFIR, Delegated Reg. 2017/565
365 chunks | Knowledge graph with provenance
""")

# --- Main area ---
st.title("Compliance-Aware RAG")
st.markdown("*Financial Regulatory Question Answering with Retrieval Validation*")

# Example questions
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

# Question input
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

    with st.spinner("Running compliance-aware pipeline..."):
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/api/ask",
                json={"question": question, "strategy": strategy},
                headers={"X-OpenAI-API-Key": api_key},
                timeout=120,
            )

            if response.status_code == 401:
                st.error("Invalid OpenAI API key. Please check your key.")
                st.stop()
            elif response.status_code != 200:
                st.error(f"Error: {response.json().get('detail', response.text)}")
                st.stop()

            result = response.json()
            elapsed = time.time() - start_time

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at {API_URL}. Is the server running?")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The pipeline may need more time for complex questions.")
            st.stop()

    # --- Display results ---

    # Status badge
    status = result["final_status"]
    if status == "ACCEPTED":
        st.success(f"ACCEPTED  |  {elapsed:.1f}s  |  Strategy: {result['strategy']}")
    else:
        st.warning(f"REJECTED  |  {elapsed:.1f}s  |  Strategy: {result['strategy']}")

    # Answer
    st.markdown("### Answer")
    st.markdown(result["answer"])

    # Validation summary
    st.markdown("### Validation")
    val = result["validation"]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Support Rate", f"{val['support_rate']:.0%}" if val["support_rate"] is not None else "N/A")
    col_b.metric("Claims", f"{val['supported_claims']}/{val['total_claims']}")
    col_c.metric("Contradictions", val["source_contradictions"])
    col_d.metric("Whole-Answer", val["whole_answer_status"])

    # Requirements
    if val.get("requirement_results"):
        st.markdown("#### Question Requirements")
        for req in val["requirement_results"]:
            icon = {"SATISFIED": "check", "MISSING": "x", "UNSUPPORTED": "warning"}.get(req["status"], "question")
            if req["status"] == "SATISFIED":
                st.markdown(f"- :white_check_mark: **{req['requirement']}** — {req['reason']}")
            elif req["status"] == "MISSING":
                st.markdown(f"- :x: **{req['requirement']}** — {req['reason']}")
            else:
                st.markdown(f"- :warning: **{req['requirement']}** — {req['reason']}")

    # Claims detail
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

    # Revision info
    if result.get("revised") and result.get("initial_validation"):
        st.markdown("#### Revision")
        iv = result["initial_validation"]
        st.info(
            f"Initial draft was **{'valid' if iv['valid'] else 'invalid'}** "
            f"(support: {iv['support_rate']}, whole-answer: {iv['whole_answer_status']}). "
            f"One revision was attempted."
        )

    # Evidence
    with st.expander("Retrieved Document Evidence", expanded=False):
        for doc in result.get("doc_evidence", []):
            st.markdown(f"**{doc['source_id']}** ({doc['doc_id']})")
            st.text(doc["text"])
            st.markdown("---")

    # Graph evidence
    with st.expander("Graph Evidence", expanded=False):
        for triple in result.get("graph_evidence", []):
            st.markdown(
                f"- `{triple['subject']}` —[{triple['relation']}]→ `{triple['object']}` "
                f"(hop={triple['depth']}, sources={', '.join(triple['source_ids'])})"
            )

    # Raw JSON
    with st.expander("Full Pipeline Output (JSON)", expanded=False):
        st.json(result)
