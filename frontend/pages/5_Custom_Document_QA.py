"""Custom Document QA — upload your own regulatory document and query it."""

import streamlit as st
import os
import sys
import time
import json
import re
import textwrap

st.set_page_config(
    page_title="Custom Document QA",
    page_icon="",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, APP_DIR)

st.title("Custom Document QA")
st.markdown(
    "*Upload your own regulatory or legal document and ask questions against it. "
    "The system will chunk it, build a vector index, optionally extract a knowledge graph, "
    "and run the full compliance-aware pipeline.*"
)

st.markdown("---")

# --- Sidebar ---
st.sidebar.title("Custom Document QA")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Required for embeddings, KG extraction, and answer generation.",
)

build_kg = st.sidebar.checkbox(
    "Build Knowledge Graph",
    value=False,
    help="Extract entities and relationships from your document. "
    "Costs ~$0.001 per chunk and adds 1-2s per chunk.",
)

chunk_size = st.sidebar.slider("Chunk size (words)", 200, 800, 400, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
1. Upload a PDF or text file
2. Document is split into chunks
3. Chunks are embedded into a FAISS vector index
4. (Optional) Knowledge graph is extracted
5. Ask questions — full pipeline runs on your document
""")

# --- Helper functions ---

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except ImportError:
        st.error("PDF support requires pdfplumber or PyPDF2. Install with: pip install pdfplumber")
        return None


def chunk_text(text, doc_id, target_words=400):
    """Split text into chunks of approximately target_words words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), target_words):
        chunk_words = words[i:i + target_words]
        chunk_text_str = " ".join(chunk_words)
        chunk_id = len(chunks)
        source_id = f"{doc_id}:chunk:{chunk_id:04d}"
        chunks.append({
            "source_id": source_id,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "filename": doc_id,
        })
    return chunks


# --- File Upload ---
st.markdown("### 1. Upload Document")

uploaded_file = st.file_uploader(
    "Upload a regulatory or legal document",
    type=["pdf", "txt"],
    help="Supported formats: PDF, plain text (.txt)",
)

if uploaded_file:
    file_name = uploaded_file.name
    doc_id = re.sub(r"[^a-zA-Z0-9]", "_", os.path.splitext(file_name)[0]).lower()

    with st.spinner("Extracting text..."):
        if file_name.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8", errors="replace")

    if not raw_text or len(raw_text.strip()) < 100:
        st.error("Could not extract sufficient text from the document. Please try a different file.")
        st.stop()

    word_count = len(raw_text.split())
    st.success(f"Extracted {word_count:,} words from **{file_name}**")

    with st.expander("Preview extracted text", expanded=False):
        st.text(raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""))

    # --- Chunking ---
    st.markdown("### 2. Processing")

    chunks = chunk_text(raw_text, doc_id, target_words=chunk_size)
    corpus_by_id = {c["source_id"]: c for c in chunks}

    st.info(f"Split into **{len(chunks)} chunks** (~{chunk_size} words each)")

    # --- Build index ---
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    # Store processing results in session state to avoid re-building
    cache_key = f"custom_doc_{doc_id}_{len(chunks)}_{build_kg}"

    if st.session_state.get("custom_cache_key") != cache_key:
        status = st.status("Building vector index and processing document...", expanded=True)

        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_core.documents import Document
            from langchain_community.vectorstores import FAISS
            from backend.core.config import OPENAI_EMBEDDING_MODEL

            # Build FAISS
            status.write(f"Embedding {len(chunks)} chunks...")
            openai_embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL, api_key=api_key,
            )
            documents = [
                Document(
                    page_content=c["text"],
                    metadata={
                        "source_id": c["source_id"],
                        "doc_id": c["doc_id"],
                        "filename": c["filename"],
                        "chunk_id": c["chunk_id"],
                    },
                )
                for c in chunks
            ]
            vectorstore = FAISS.from_documents(documents, openai_embeddings)
            status.write(f"Vector index built with {len(chunks)} chunks.")

            # Build KG (optional)
            custom_G = None
            custom_node_list = None
            custom_node_embeddings = None

            if build_kg:
                import spacy
                from sentence_transformers import SentenceTransformer
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import PromptTemplate
                from backend.core.config import RELATION_TYPES, LOCAL_EMBEDDING_MODEL
                from backend.core.graph import (
                    build_graph_from_corpus, embed_graph_nodes,
                    canonicalise_relation,
                )

                status.write("Loading NLP models for KG extraction...")
                nlp = spacy.load("en_core_web_sm")
                embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

                kg_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

                extraction_prompt = PromptTemplate.from_template(
                    """You are extracting a regulatory knowledge graph from a legal document.
Extract regulatory entities and directed relationships from the text.
For every relationship choose EXACTLY ONE label from:
{relation_types}

Return ONLY valid JSON:
{{"entities": ["entity"], "relationships": [["subject", "RELATION", "object"]]}}

Rules:
- Do not invent facts not explicit in the text.
- Prefer concise canonical entity names.
- Relationship subjects and objects must be included as entities.
- Use RELATED_TO only if no more specific label is justified.

Text:
{text}
"""
                )

                def extract_elements(chunk_text_inner):
                    try:
                        response = kg_llm.invoke(
                            extraction_prompt.format(
                                relation_types=", ".join(RELATION_TYPES),
                                text=chunk_text_inner,
                            )
                        )
                        raw = re.sub(r"```json|```", "", response.content.strip()).strip()
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            match = re.search(r"\{.*\}", raw, re.DOTALL)
                            parsed = json.loads(match.group()) if match else {"entities": [], "relationships": []}
                        entities = [str(x) for x in parsed.get("entities", [])]
                        relationships = []
                        for rel in parsed.get("relationships", []):
                            if isinstance(rel, list) and len(rel) == 3:
                                relationships.append([rel[0], canonicalise_relation(rel[1]), rel[2]])
                        return {"entities": entities, "relationships": relationships}
                    except Exception:
                        return {"entities": [], "relationships": []}

                status.write(f"Extracting knowledge graph from {len(chunks)} chunks...")
                progress_bar = status.progress(0)
                # Build graph chunk by chunk with progress
                import networkx as nx
                from backend.core.graph import normalise_entity

                custom_G = nx.DiGraph()

                def _add_node_source(node, source_id):
                    if not custom_G.has_node(node):
                        custom_G.add_node(node, source_ids=[])
                    ids = custom_G.nodes[node].setdefault("source_ids", [])
                    if source_id not in ids:
                        ids.append(source_id)

                def _add_edge(subject, relation, obj, source_id):
                    if custom_G.has_edge(subject, obj):
                        data = custom_G[subject][obj]
                        labels = set(data.get("labels", []))
                        labels.add(relation)
                        data["labels"] = sorted(labels)
                        ids = data.setdefault("source_ids", [])
                        if source_id not in ids:
                            ids.append(source_id)
                    else:
                        custom_G.add_edge(subject, obj, labels=[relation], source_ids=[source_id])

                for idx, item in enumerate(chunks):
                    elements = extract_elements(item["text"])
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
                    progress_bar.progress((idx + 1) / len(chunks))

                status.write(
                    f"Knowledge graph: {custom_G.number_of_nodes()} nodes, "
                    f"{custom_G.number_of_edges()} edges"
                )
                custom_node_list, custom_node_embeddings = embed_graph_nodes(custom_G, embedder)
            else:
                # Load shared NLP models for validation (needed even without KG)
                import spacy
                from sentence_transformers import SentenceTransformer
                import numpy as np
                from backend.core.config import LOCAL_EMBEDDING_MODEL

                nlp = spacy.load("en_core_web_sm")
                embedder = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

                import networkx as nx
                custom_G = nx.DiGraph()
                custom_node_list = []
                custom_node_embeddings = np.zeros((0, 384))

            # Save to session state
            st.session_state["custom_cache_key"] = cache_key
            st.session_state["custom_vectorstore"] = vectorstore
            st.session_state["custom_corpus"] = chunks
            st.session_state["custom_corpus_by_id"] = corpus_by_id
            st.session_state["custom_G"] = custom_G
            st.session_state["custom_node_list"] = custom_node_list
            st.session_state["custom_node_embeddings"] = custom_node_embeddings
            st.session_state["custom_embedder"] = embedder
            st.session_state["custom_nlp"] = nlp
            st.session_state["custom_doc_name"] = file_name

            status.update(label="Document processed successfully!", state="complete", expanded=False)

        except Exception as e:
            error_msg = str(e)
            status.update(label="Processing failed", state="error", expanded=False)
            if "Incorrect API key" in error_msg or "api key" in error_msg.lower():
                st.error("Invalid OpenAI API key. Please check your key.")
            else:
                st.error(f"Processing error: {error_msg}")
            st.stop()

    # --- Query interface ---
    st.markdown("### 3. Ask Questions")

    if st.session_state.get("custom_cache_key") == cache_key:
        doc_name = st.session_state.get("custom_doc_name", file_name)
        custom_G = st.session_state["custom_G"]
        has_kg = custom_G.number_of_nodes() > 0

        if has_kg:
            st.info(
                f"Ready to query **{doc_name}** — "
                f"{len(chunks)} chunks, "
                f"{custom_G.number_of_nodes()} graph nodes, "
                f"{custom_G.number_of_edges()} graph edges"
            )
        else:
            st.info(f"Ready to query **{doc_name}** — {len(chunks)} chunks (vector-only mode)")

        question = st.text_area(
            "Ask a question about your document:",
            height=80,
            placeholder="e.g., What are the main obligations described in this document?",
        )

        submit = st.button("Ask", type="primary")

        if submit:
            if not question or len(question.strip()) < 10:
                st.error("Please enter a question (at least 10 characters).")
                st.stop()

            status_container = st.status("Running pipeline on your document...", expanded=True)

            def on_step(name, state, info):
                if state == "running":
                    status_container.write(f"**{name}** — running...")
                elif state == "done" and info:
                    dur = info.get("duration_s", 0)
                    detail = info.get("detail", "")
                    status_container.write(f"**{name}** — {dur:.1f}s — {detail}")

            start_time = time.time()
            try:
                from backend.core.pipeline import CompliancePipeline
                from backend.core.corpus import build_article_index

                article_index, _ = build_article_index(chunks)

                pipeline = CompliancePipeline(
                    api_key=api_key,
                    corpus=st.session_state["custom_corpus"],
                    corpus_by_id=st.session_state["custom_corpus_by_id"],
                    article_index=article_index,
                    G=st.session_state["custom_G"],
                    node_list=st.session_state["custom_node_list"],
                    node_embeddings=st.session_state["custom_node_embeddings"],
                    embedder=st.session_state["custom_embedder"],
                    nlp=st.session_state["custom_nlp"],
                    vectorstore=st.session_state["custom_vectorstore"],
                )

                use_strategy = "legal_hybrid" if has_kg else "legal_vector"
                result = pipeline.run(question, strategy=use_strategy, on_step=on_step)
                elapsed = time.time() - start_time

                status_container.update(
                    label=f"Pipeline complete — {elapsed:.1f}s",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                error_msg = str(e)
                status_container.update(label="Pipeline failed", state="error", expanded=False)
                st.error(f"Pipeline error: {error_msg}")
                st.stop()

            # --- Display results ---
            final_status = result["final_status"]
            if final_status == "ACCEPTED":
                st.success(f"ACCEPTED  |  {elapsed:.1f}s")
            else:
                st.warning(f"REJECTED  |  {elapsed:.1f}s")

            st.markdown("### Answer")

            def _stream_answer(text):
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(0.02)

            st.write_stream(_stream_answer(result["answer"]))

            # Validation summary
            st.markdown("### Validation")
            val = result["validation"]
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Support Rate", f"{val['support_rate']:.0%}" if val["support_rate"] is not None else "N/A")
            col_b.metric("Claims", f"{val['supported_claims']}/{val['total_claims']}")
            col_c.metric("Contradictions", val["source_contradictions"])
            col_d.metric("Whole-Answer", val["whole_answer_status"])

            if val.get("claims"):
                with st.expander("Claim Verification", expanded=False):
                    for claim in val["claims"]:
                        status_icon = {
                            "SUPPORTED_SOURCE": ":white_check_mark:",
                            "SUPPORTED_GRAPH": ":large_blue_circle:",
                            "CONTRADICTED_SOURCE": ":red_circle:",
                            "GRAPH_CONFLICT": ":red_circle:",
                            "UNVERIFIED": ":yellow_circle:",
                        }.get(claim["final_status"], ":question:")
                        st.markdown(
                            f"{status_icon} `{claim['subject']}` —[{claim['relation']}]→ "
                            f"`{claim['object']}` : **{claim['final_status']}**"
                        )

            with st.expander("Retrieved Evidence", expanded=False):
                for doc in result.get("doc_evidence", []):
                    st.markdown(f"**{doc['source_id']}**")
                    st.text(doc["text"])
                    st.markdown("---")

            with st.expander("Full Pipeline Output (JSON)", expanded=False):
                st.json(result)
