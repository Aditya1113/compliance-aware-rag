"""Configuration constants extracted from the dissertation notebook."""

import os

# Paths — APP_ROOT is the app/ directory
APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
# Data can be in app/data/ (deployed) or project_root/data/ (local dev)
DATA_DIR = os.path.join(APP_ROOT, "data") if os.path.isdir(os.path.join(APP_ROOT, "data")) else os.path.join(PROJECT_ROOT, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus_frozen_2026-08-11.json")
GRAPH_CACHE_PATH = os.path.join(DATA_DIR, "graph_frozen.pkl")
FAISS_CACHE_DIR = os.path.join(DATA_DIR, "faiss_index")

# Models
GENERATION_MODEL = "gpt-4o-mini"
CLAIM_EXTRACTION_MODEL = "gpt-4o-mini"
SOURCE_VERIFIER_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retrieval
RETRIEVAL_EVAL_KS = (3, 5, 10)
VECTOR_CANDIDATE_K = 10
GENERATION_DOC_K = 5
GRAPH_GENERATION_TOP_N = 10
RRF_CONSTANT = 60

# Graph search (DEV-selected)
SELECTED_GRAPH_SEED_K = 5
SELECTED_GRAPH_HOPS = 2
SELECTED_GRAPH_TRIPLE_TOP_N = 10
VECTOR_RRF_WEIGHT = 4.0
GRAPH_RRF_WEIGHT = 1.0
EXCLUDED_GRAPH_RELATIONS = {"RELATED_TO"}

# Validation
FUZZY_MATCH_THRESHOLD = 0.72
VALIDATION_THRESHOLD = 0.80
REQUIRE_ALL_QUESTION_PARTS = True
MAX_ATTEMPTS = 2

# Relation types for KG
RELATION_TYPES = [
    "REQUIRES", "PROHIBITS", "PERMITS", "GOVERNS", "APPLIES_TO",
    "HAS_OBLIGATION", "SUPERVISES", "REPORTS_TO", "AUTHORIZES",
    "ENSURES", "EXEMPTS", "DEFINES", "SUBJECT_TO", "DISCLOSES_TO",
    "NOTIFIES", "COOPERATES_WITH", "RELATED_TO",
]
RELATION_SET = set(RELATION_TYPES)

# Contradictory relation pairs for symbolic checking
CONTRADICTORY_RELATIONS = {
    "REQUIRES": {"EXEMPTS", "PROHIBITS"},
    "PERMITS": {"PROHIBITS"},
    "PROHIBITS": {"PERMITS", "AUTHORIZES"},
    "AUTHORIZES": {"PROHIBITS"},
    "HAS_OBLIGATION": {"EXEMPTS"},
    "SUBJECT_TO": {"EXEMPTS"},
}
