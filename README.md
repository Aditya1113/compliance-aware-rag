# Compliance-Aware RAG for Financial Regulatory QA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://compliance-aware-rag.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)](https://python.langchain.com/)
[![License: Academic](https://img.shields.io/badge/license-academic-lightgrey.svg)](#license)

> A RAG system that doesn't just retrieve and generate -- it validates every claim, checks legal attribution, and rejects its own answer when it can't verify correctness.

A production-grade compliance question-answering system that combines knowledge-graph-augmented retrieval with post-generation validation to answer financial regulatory questions grounded in EU legislation (MiFID II, MiFIR, Delegated Regulation 2017/565).

Built as an MSc dissertation project at the University of Edinburgh and refactored into a deployable web application.

**[Try the live demo](https://compliance-aware-rag.streamlit.app)** (bring your own OpenAI API key -- ~$0.02 per query)

## What Makes This Different

Most RAG systems retrieve relevant text and generate an answer. This system goes further:

1. **Knowledge Graph Augmentation** — A provenance-linked regulatory knowledge graph captures inter-provision relationships (e.g., which entity supervises which, what exemptions apply where). Graph triples supplement dense vector retrieval.

2. **Legal-Aware Retrieval** — An Article-heading resolver handles explicit regulatory citations (e.g., "Article 16(5)") that dense similarity search misses, combined with weighted reciprocal rank fusion across 4 retrieval strategies.

3. **Post-Generation Validation** — Every answer is decomposed into atomic claims that are verified against both the source text and the knowledge graph. A whole-answer legal-attribution check catches misattributed duties that pass claim-level checks.

4. **Agentic Revision Loop** — If validation fails, one bounded revision is attempted with structured feedback before a final accept/reject decision.

## Architecture

```
User Question
     |
     v
[Retrieval Layer]
  |-- Dense Vector (FAISS + OpenAI embeddings)
  |-- Legal-aware Vector (+ Article resolver)
  |-- Graph-only (BFS + semantic reranking)
  |-- Legal-aware Hybrid (weighted RRF fusion)  <-- default
     |
     v
[Generation Layer]
  |-- Question decomposition into requirements
  |-- Evidence-constrained answer drafting (GPT-4o-mini)
     |
     v
[Validation Layer]
  |-- Atomic claim extraction
  |-- Source-grounded claim verification
  |-- Symbolic KG consistency checking
  |-- Whole-answer legal-attribution verification
  |-- Question-completeness checking
     |
     v
[Routing Layer]
  |-- ACCEPT --> return answer with audit trail
  |-- REVISE --> structured feedback --> re-draft --> re-validate
  |-- REJECT --> flag answer as unreliable
```

## App Pages

| Page | Description |
|------|-------------|
| **Compliance QA** | Main question-answering interface with full pipeline execution, validation display, and pipeline trace timeline |
| **Knowledge Graph Explorer** | Interactive visualization of the regulatory knowledge graph -- browse entities, relationships, and explore neighbourhoods with pyvis |
| **Retrieval Strategy Comparison** | Run all 4 retrieval strategies side-by-side on the same query -- overlap matrix, unique contributions, ranked results |

## Key Results (from dissertation evaluation)

| Metric | Value |
|--------|-------|
| Legal-aware Hybrid Recall@5 | 0.82 |
| Pipeline acceptance rate | 62% (31/50) |
| Correct rejections | 84% of rejected answers had quality issues |
| Cross-model agreement | 88% (GPT-4o vs Claude Sonnet) |
| Knowledge graph | ~2,400 nodes, ~2,900 edges, 17 relation types |
| LLM cost per query | ~$0.02 (GPT-4o-mini) |

## Tech Stack

- **Backend**: FastAPI, LangChain, LangGraph, NetworkX, FAISS, spaCy
- **Frontend**: Streamlit
- **LLMs**: GPT-4o-mini (generation, validation, claim extraction)
- **Embeddings**: OpenAI text-embedding-3-small + all-MiniLM-L6-v2
- **Corpus**: 365 chunks from 3 EU regulatory instruments

## Screenshots

### Compliance QA — Pipeline Trace + Validated Answer
![Pipeline trace showing step-by-step timing, followed by the validated answer with claim verification](docs/screenshots/qa_pipeline.png)

### Knowledge Graph Explorer — Interactive Visualization
![Interactive pyvis graph showing regulatory entity relationships with colour-coded relation types](docs/screenshots/kg_explorer.png)

### Retrieval Strategy Comparison — Overlap Analysis
![Side-by-side ranking comparison with overlap matrix showing how strategies complement each other](docs/screenshots/retrieval_comparison.png)

> **Note:** Replace the screenshot paths above with actual screenshots of your running app. To capture them:
> 1. Run the app locally with `streamlit run frontend/streamlit_app.py`
> 2. Take screenshots of each page
> 3. Save them in `docs/screenshots/`

## Quick Start

### Prerequisites
- Python 3.11+
- An OpenAI API key

### Setup

```bash
cd app
chmod +x setup.sh
export OPENAI_API_KEY=sk-your-key-here
./setup.sh
```

### Run

```bash
# Terminal 1 — Backend
cd app && source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd app && source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```

Open **http://localhost:8501** and enter your OpenAI API key in the sidebar.

### First-Time Data Setup

The corpus is included. The knowledge graph and FAISS index need to be built once:

```bash
# Build knowledge graph (~365 API calls to gpt-4o-mini, ~$0.50)
cd app/backend && python build_graph.py

# Build FAISS index (~365 embedding calls, ~$0.01)
cd app/backend && python build_index.py
```

## BYOK (Bring Your Own Key)

Users enter their own OpenAI API key in the browser. The key is:
- Sent per-request in the `X-OpenAI-API-Key` header
- Never stored server-side
- Never logged

This means zero LLM hosting cost for the operator.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | System status and corpus stats |
| `GET` | `/api/strategies` | List available retrieval strategies |
| `GET` | `/api/graph/stats` | Knowledge graph statistics |
| `POST` | `/api/ask` | Submit a question (requires `X-OpenAI-API-Key` header) |

### Example API Call

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -H "X-OpenAI-API-Key: sk-your-key" \
  -d '{"question": "Under MiFID II Article 16(5), what are the responsibilities of the compliance function?", "strategy": "legal_hybrid"}'
```

## Project Structure

```
app/
  backend/
    core/
      config.py         # All configuration constants
      corpus.py         # Corpus loading and article index
      graph.py          # Knowledge graph operations and symbolic checking
      retrieval.py      # 4 retrieval strategies
      generation.py     # Answer generation and revision
      validation.py     # Post-generation validation pipeline
      pipeline.py       # Full end-to-end pipeline orchestration
    api/
      routes.py         # FastAPI endpoints
    main.py             # FastAPI app with startup resource loading
    build_graph.py      # One-time graph construction script
    build_index.py      # One-time FAISS index construction script
  frontend/
    streamlit_app.py                          # Main QA page
    pages/
      2_Knowledge_Graph_Explorer.py           # Interactive KG visualization
      3_Retrieval_Strategy_Comparison.py      # Side-by-side retrieval comparison
  requirements.txt
  Dockerfile
  setup.sh
  README.md
```

## Dissertation

This application is based on the MSc dissertation:

> **Compliance-Aware RAG for Financial Regulatory Question Answering: Retrieval and Validation**
> Aditya Mazumdar, University of Edinburgh, 2026

The dissertation source, benchmarks, and experimental results are in the parent directory.

## License

Academic project. Contact the author for licensing.
