"""Full compliance-aware RAG pipeline: retrieve -> draft -> validate -> revise/accept/reject."""

import copy
import time

from langchain_openai import OpenAIEmbeddings

from .config import (
    GENERATION_DOC_K, MAX_ATTEMPTS,
    CLAIM_EXTRACTION_MODEL, SOURCE_VERIFIER_MODEL,
    OPENAI_EMBEDDING_MODEL,
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
        # Swap the vectorstore's embedding function to use the user's API key
        self.vectorstore.embedding_function = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL, api_key=api_key,
        )

        self.generation_llm = create_llm(api_key)
        self.claim_llm = create_llm(api_key, CLAIM_EXTRACTION_MODEL)
        self.verifier_llm = create_llm(api_key, SOURCE_VERIFIER_MODEL)

    def run(self, query, strategy="legal_hybrid"):
        trace = []

        def _step(name):
            return {"step": name, "start": time.time()}

        def _end(step_info, **extra):
            step_info["duration_s"] = round(time.time() - step_info["start"], 2)
            step_info.update(extra)
            del step_info["start"]
            trace.append(step_info)

        # 1. Retrieve
        s = _step("Retrieval")
        rankings = retrieve_all_strategies(
            query, self.vectorstore, self.article_index,
            self.G, self.node_list, self.node_embeddings,
            self.embedder, self.corpus_by_id,
        )
        selected_rank = rankings.get(strategy, rankings["legal_hybrid"])
        doc_ids = selected_rank[:GENERATION_DOC_K]
        docs = docs_from_source_ids(doc_ids, self.corpus_by_id)
        graph_evidence = rankings["graph_evidence"]
        _end(s, detail=f"{len(doc_ids)} docs, {len(graph_evidence)} graph triples")

        # 2. Extract requirements
        s = _step("Question Decomposition")
        requirements = extract_question_requirements(query, self.verifier_llm)
        _end(s, detail=f"{len(requirements)} requirements extracted")

        # 3. Generate draft
        s = _step("Answer Generation")
        answer, gen_latency = generate_draft(
            self.generation_llm, docs, graph_evidence, query, requirements,
        )
        initial_answer = answer
        _end(s, detail=f"{len(answer.split())} words generated")

        # 4. Validate
        s = _step("Validation")
        validation = validate_answer(
            query, answer, docs, requirements,
            self.G, self.node_list, self.node_embeddings,
            self.embedder, self.nlp,
            self.claim_llm, self.verifier_llm,
        )
        initial_validation = copy.deepcopy(validation)
        _end(s, detail=(
            f"{validation['total_claims']} claims, "
            f"{validation['supported_claims']} supported, "
            f"{validation['source_contradictions']} contradictions, "
            f"whole-answer: {validation['whole_answer_verification']['status']}"
        ))

        # 5. Route: accept / revise / reject
        attempts = 1
        final_status = None
        revised = False

        if validation["valid"]:
            final_status = "ACCEPTED"
            trace.append({"step": "Routing", "duration_s": 0, "detail": "ACCEPTED"})
        elif attempts < MAX_ATTEMPTS:
            trace.append({"step": "Routing", "duration_s": 0, "detail": "Validation failed -> REVISE"})

            # Revise
            s = _step("Revision")
            answer, rev_latency = generate_revision(
                self.generation_llm, docs, graph_evidence,
                query, requirements, answer, validation,
            )
            gen_latency += rev_latency
            revised = True
            _end(s, detail=f"{len(answer.split())} words in revised answer")

            # Re-validate
            s = _step("Re-validation")
            validation = validate_answer(
                query, answer, docs, requirements,
                self.G, self.node_list, self.node_embeddings,
                self.embedder, self.nlp,
                self.claim_llm, self.verifier_llm,
            )
            attempts += 1
            final_status = "ACCEPTED" if validation["valid"] else "REJECTED"
            _end(s, detail=(
                f"{validation['total_claims']} claims, "
                f"{validation['supported_claims']} supported -> {final_status}"
            ))
        else:
            final_status = "REJECTED"
            trace.append({"step": "Routing", "duration_s": 0, "detail": "REJECTED (max attempts)"})

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
            "trace": trace,
        }
