"""Full compliance-aware RAG pipeline: retrieve -> draft -> validate -> revise/accept/reject."""

import copy

from .config import (
    GENERATION_DOC_K, MAX_ATTEMPTS,
    CLAIM_EXTRACTION_MODEL, SOURCE_VERIFIER_MODEL,
)
from .corpus import docs_from_source_ids
from .generation import (
    create_llm, generate_draft, generate_revision, is_answered,
)
from .retrieval import retrieve_all_strategies
from .validation import (
    extract_question_requirements, validate_answer,
)


class CompliancePipeline:
    """Encapsulates the full compliance-aware RAG pipeline with BYOK."""

    def __init__(self, api_key, corpus, corpus_by_id, article_index,
                 G, node_list, node_embeddings, embedder, nlp, vectorstore):
        self.api_key = api_key
        self.corpus = corpus
        self.corpus_by_id = corpus_by_id
        self.article_index = article_index
        self.G = G
        self.node_list = node_list
        self.node_embeddings = node_embeddings
        self.embedder = embedder
        self.nlp = nlp
        self.vectorstore = vectorstore

        self.generation_llm = create_llm(api_key)
        self.claim_llm = create_llm(api_key, CLAIM_EXTRACTION_MODEL)
        self.verifier_llm = create_llm(api_key, SOURCE_VERIFIER_MODEL)

    def run(self, query, strategy="legal_hybrid"):
        # 1. Retrieve
        rankings = retrieve_all_strategies(
            query, self.vectorstore, self.article_index,
            self.G, self.node_list, self.node_embeddings,
            self.embedder, self.corpus_by_id,
        )

        selected_rank = rankings.get(strategy, rankings["legal_hybrid"])
        doc_ids = selected_rank[:GENERATION_DOC_K]
        docs = docs_from_source_ids(doc_ids, self.corpus_by_id)
        graph_evidence = rankings["graph_evidence"]

        # 2. Extract requirements
        requirements = extract_question_requirements(query, self.verifier_llm)

        # 3. Generate draft
        answer, gen_latency = generate_draft(
            self.generation_llm, docs, graph_evidence, query, requirements,
        )
        initial_answer = answer

        # 4. Validate
        validation = validate_answer(
            query, answer, docs, requirements,
            self.G, self.node_list, self.node_embeddings,
            self.embedder, self.nlp,
            self.claim_llm, self.verifier_llm,
        )
        initial_validation = copy.deepcopy(validation)

        # 5. Route: accept / revise / reject
        attempts = 1
        final_status = None
        revised = False

        if validation["valid"]:
            final_status = "ACCEPTED"
        elif attempts < MAX_ATTEMPTS:
            # Revise
            answer, rev_latency = generate_revision(
                self.generation_llm, docs, graph_evidence,
                query, requirements, answer, validation,
            )
            gen_latency += rev_latency
            revised = True

            # Re-validate
            validation = validate_answer(
                query, answer, docs, requirements,
                self.G, self.node_list, self.node_embeddings,
                self.embedder, self.nlp,
                self.claim_llm, self.verifier_llm,
            )
            attempts += 1
            final_status = "ACCEPTED" if validation["valid"] else "REJECTED"
        else:
            final_status = "REJECTED"

        return {
            "query": query,
            "strategy": strategy,
            "answer": answer,
            "final_status": final_status,
            "revised": revised,
            "initial_answer": initial_answer,
            "attempts": attempts,
            "generation_latency_s": gen_latency,
            "retrieval": {
                "strategy_used": strategy,
                "doc_ids": doc_ids,
                "rankings": {k: v for k, v in rankings.items()
                             if k not in ("graph_evidence", "seed_nodes")},
            },
            "graph_evidence": [
                {
                    "subject": e["subject"],
                    "relation": e["relation"],
                    "object": e["object"],
                    "depth": e["depth"],
                    "source_ids": e["source_ids"],
                }
                for e in graph_evidence
            ],
            "doc_evidence": [
                {"source_id": d["source_id"], "doc_id": d["doc_id"], "text": d["text"][:300] + "..."}
                for d in docs
            ],
            "requirements": requirements,
            "validation": {
                "valid": validation["valid"],
                "support_rate": validation["support_rate"],
                "total_claims": validation["total_claims"],
                "supported_claims": validation["supported_claims"],
                "source_contradictions": validation["source_contradictions"],
                "graph_conflicts": validation["graph_conflicts"],
                "whole_answer_status": validation["whole_answer_verification"]["status"],
                "whole_answer_reason": validation["whole_answer_verification"]["reason"],
                "requirements_satisfied": validation["requirements_satisfied"],
                "requirement_results": validation["requirement_results"],
                "claims": [
                    {
                        "subject": c["subject"],
                        "relation": c["relation"],
                        "object": c["object"],
                        "final_status": c["final_status"],
                    }
                    for c in validation["all_claims"]
                ],
            },
            "initial_validation": {
                "valid": initial_validation["valid"],
                "support_rate": initial_validation["support_rate"],
                "whole_answer_status": initial_validation["whole_answer_verification"]["status"],
            } if revised else None,
        }
