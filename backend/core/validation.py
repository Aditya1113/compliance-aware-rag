"""Post-generation validation: claim extraction, source verification, symbolic checking."""

import json
import re

from langchain_core.prompts import PromptTemplate

from .config import (
    CLAIM_EXTRACTION_MODEL, SOURCE_VERIFIER_MODEL,
    RELATION_TYPES, VALIDATION_THRESHOLD, REQUIRE_ALL_QUESTION_PARTS,
)
from .corpus import format_doc_evidence
from .generation import (
    create_llm, invoke_with_metadata, parse_json_object, is_answered,
)
from .graph import canonicalise_relation, check_claim_symbolically


# --- Claim extraction ---

CLAIM_EXTRACTION_PROMPT = PromptTemplate.from_template("""Extract atomic regulatory claims from the answer as subject-relation-object triples.

Choose the relation EXACTLY from:
{relation_types}

Return ONLY valid JSON:
{{"claims": [
  {{"subject": "...", "relation": "REQUIRES", "object": "..."}}
]}}

Rules:
- Extract only claims actually stated in the answer.
- Split compound assertions into separate claims where practical.
- Do not add external knowledge.

Answer:
{answer}
""")


def extract_claims(answer_text, claim_llm):
    if not is_answered(answer_text):
        return []
    response = claim_llm.invoke(
        CLAIM_EXTRACTION_PROMPT.format(
            relation_types=", ".join(RELATION_TYPES),
            answer=answer_text,
        )
    )
    parsed = parse_json_object(response.content, {"claims": []})
    claims = []
    for claim in parsed.get("claims", []):
        if all(key in claim for key in ("subject", "relation", "object")):
            claims.append({
                "subject": str(claim["subject"]),
                "relation": canonicalise_relation(claim["relation"]),
                "object": str(claim["object"]),
            })
    return claims


# --- Question requirements ---

def _repair_requirement_fragments(requirements):
    repaired = []
    fragment_prefixes = ("of ", "for ", "under ", "in ", "by ", "from ", "with ")
    interrogative_pattern = re.compile(
        r"^(of|for|under|in|by|from|with)\s+(what|when|whom|who|which|where|how)\b",
        re.IGNORECASE,
    )
    for requirement in requirements:
        req = requirement.strip()
        lower_req = req.lower()
        looks_like_fragment = lower_req.startswith(fragment_prefixes)
        is_complete = bool(interrogative_pattern.match(req))
        if repaired and looks_like_fragment and not is_complete:
            repaired[-1] = repaired[-1].rstrip(" ?") + " " + req
        else:
            repaired.append(req)
    return repaired


def extract_question_requirements(question, verifier_llm):
    prompt = f"""You are decomposing a financial-regulatory question into the minimum number of EXPLICIT answer requirements.

QUESTION:
{question}

Rules:
1. Each requirement must be a COMPLETE, SELF-CONTAINED question or task.
2. Preserve the level of granularity used by the original question.
3. Never invent implied requirements.
4. Never split a noun phrase, prepositional phrase, subject, object, qualifier, or legal citation into a separate requirement.
5. Split only when the wording explicitly asks for multiple distinct answer slots.
6. The number of requirements must not exceed the number of distinct pieces of information explicitly requested.

Return ONLY valid JSON:
{{
  "requirements": ["..."]
}}
"""
    call = invoke_with_metadata(verifier_llm, prompt)
    parsed = parse_json_object(call["content"], {"requirements": []})
    requirements = [str(r).strip() for r in parsed.get("requirements", []) if str(r).strip()]
    if not requirements:
        requirements = [question.strip()]
    return _repair_requirement_fragments(requirements)


# --- Whole-answer verification ---

def verify_whole_answer(query, answer, docs, verifier_llm):
    prompt = f"""You are a strict legal-grounding verifier for EU financial regulation.

QUESTION:
{query}

ANSWER TO VERIFY:
{answer}

AUTHORITATIVE REGULATORY EVIDENCE:
{format_doc_evidence(docs)}

Classify the WHOLE ANSWER as exactly one of:

SUPPORTED - The material claims are supported by the regulatory evidence and duties are attributed to the correct legal actor.
CONTRADICTED - At least one material claim conflicts with the evidence, or the answer attributes a duty to the wrong legal actor.
NOT_ENOUGH_EVIDENCE - The evidence is insufficient to determine correctness.

IMPORTANT: Legal actor alignment is mandatory. A duty imposed on an "investment firm" does NOT support a claim that the same duty is a responsibility of the firm's "compliance function".

Return ONLY valid JSON:
{{
    "status": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_EVIDENCE",
    "reason": "A concise explanation."
}}
"""
    call = invoke_with_metadata(verifier_llm, prompt)
    parsed = parse_json_object(call["content"], {"status": "NOT_ENOUGH_EVIDENCE", "reason": "No valid verifier output."})
    status = str(parsed.get("status", "NOT_ENOUGH_EVIDENCE")).upper()
    if status not in {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"}:
        status = "NOT_ENOUGH_EVIDENCE"
    return {"status": status, "reason": parsed.get("reason", ""), "latency_s": call["latency_s"]}


# --- Source-grounded verification ---

SOURCE_VALIDATION_PROMPT = PromptTemplate.from_template("""You are a strict evidence-grounding verifier for EU financial regulation.

AUTHORITATIVE REGULATORY EVIDENCE:
{doc_context}

QUESTION:
{question}

ANSWER:
{answer}

ATOMIC CLAIMS:
{claims_json}

EXPLICIT QUESTION REQUIREMENTS:
{requirements_json}

For EACH atomic claim, decide: SUPPORTED, CONTRADICTED, or NOT_ENOUGH_EVIDENCE.
For EACH requirement, decide: SATISFIED, MISSING, or UNSUPPORTED.

Rules:
- Use ONLY the provided regulatory evidence.
- Legal actor alignment is mandatory.
- A duty imposed on an "investment firm" does NOT support a claim that the same duty is a responsibility of the firm's "compliance function".
- evidence_source_ids must contain only source IDs shown in the evidence.

Return ONLY valid JSON:
{{
  "claim_results": [{{"claim_index": 0, "status": "SUPPORTED", "evidence_source_ids": [], "reason": "..."}}],
  "requirement_results": [{{"requirement_index": 0, "status": "SATISFIED", "evidence_source_ids": [], "reason": "..."}}]
}}
""")


def verify_against_sources(question, answer, claims, requirements, docs, verifier_llm):
    if not docs:
        return {
            "claim_results": [],
            "requirement_results": [
                {"requirement_index": i, "status": "MISSING", "evidence_source_ids": [], "reason": "No evidence."}
                for i in range(len(requirements))
            ],
        }

    allowed = {doc["source_id"] for doc in docs}
    response = verifier_llm.invoke(
        SOURCE_VALIDATION_PROMPT.format(
            doc_context=format_doc_evidence(docs),
            question=question, answer=answer,
            claims_json=json.dumps(claims, ensure_ascii=False),
            requirements_json=json.dumps(requirements, ensure_ascii=False),
        )
    )

    parsed = parse_json_object(response.content, {"claim_results": [], "requirement_results": []})

    claim_by_idx = {int(x.get("claim_index")): x for x in parsed.get("claim_results", []) if str(x.get("claim_index", "")).isdigit()}
    req_by_idx = {int(x.get("requirement_index")): x for x in parsed.get("requirement_results", []) if str(x.get("requirement_index", "")).isdigit()}

    claim_results = []
    for i in range(len(claims)):
        r = claim_by_idx.get(i, {})
        status = str(r.get("status", "NOT_ENOUGH_EVIDENCE")).upper()
        if status not in {"SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_EVIDENCE"}:
            status = "NOT_ENOUGH_EVIDENCE"
        claim_results.append({
            "claim_index": i, "status": status,
            "evidence_source_ids": [sid for sid in r.get("evidence_source_ids", []) if sid in allowed],
            "reason": str(r.get("reason", "")),
        })

    requirement_results = []
    for i in range(len(requirements)):
        r = req_by_idx.get(i, {})
        status = str(r.get("status", "MISSING")).upper()
        if status not in {"SATISFIED", "MISSING", "UNSUPPORTED"}:
            status = "MISSING"
        requirement_results.append({
            "requirement_index": i, "requirement": requirements[i],
            "status": status,
            "evidence_source_ids": [sid for sid in r.get("evidence_source_ids", []) if sid in allowed],
            "reason": str(r.get("reason", "")),
        })

    return {"claim_results": claim_results, "requirement_results": requirement_results}


# --- Full validation ---

def validate_answer(query, answer, docs, requirements, G, node_list, node_embeddings,
                    embedder, nlp, claim_llm, verifier_llm,
                    threshold=VALIDATION_THRESHOLD):
    if not is_answered(answer):
        return {
            "valid": False, "support_rate": None, "total_claims": 0,
            "supported_claims": 0, "source_contradictions": 0, "graph_conflicts": 0,
            "whole_answer_verification": {"status": "NOT_ENOUGH_EVIDENCE", "reason": "No substantive answer."},
            "requirements": requirements,
            "requirements_satisfied": False,
            "requirement_results": [
                {"requirement_index": i, "requirement": r, "status": "MISSING", "evidence_source_ids": [], "reason": "No answer."}
                for i, r in enumerate(requirements)
            ],
            "evidence_refs": [], "all_claims": [],
            "reason": "No substantive answer was produced.",
        }

    claims = extract_claims(answer, claim_llm)

    symbolic_results = [
        check_claim_symbolically(G, c["subject"], c["relation"], c["object"],
                                 node_list, node_embeddings, embedder, nlp)
        for c in claims
    ]

    source_report = verify_against_sources(query, answer, claims, requirements, docs, verifier_llm)
    whole_answer = verify_whole_answer(query, answer, docs, verifier_llm)
    whole_contradicted = whole_answer["status"] == "CONTRADICTED"

    combined_claims = []
    for i, claim in enumerate(claims):
        symbolic = symbolic_results[i]
        source = source_report["claim_results"][i] if i < len(source_report["claim_results"]) else {
            "status": "NOT_ENOUGH_EVIDENCE", "evidence_source_ids": [], "reason": ""
        }
        source_status = source["status"]
        symbolic_status = symbolic["status"]

        if source_status == "SUPPORTED":
            final_status = "SUPPORTED_SOURCE"
        elif source_status == "CONTRADICTED":
            final_status = "CONTRADICTED_SOURCE"
        elif symbolic_status == "SUPPORTED":
            final_status = "SUPPORTED_GRAPH"
        elif symbolic_status == "VIOLATION":
            final_status = "GRAPH_CONFLICT"
        else:
            final_status = "UNVERIFIED"

        combined_claims.append({
            **claim, "final_status": final_status,
            "source_check": source, "symbolic_check": symbolic,
        })

    supported = sum(c["final_status"] in {"SUPPORTED_SOURCE", "SUPPORTED_GRAPH"} for c in combined_claims)
    source_contradictions = sum(c["final_status"] == "CONTRADICTED_SOURCE" for c in combined_claims)
    graph_conflicts = sum(c["final_status"] == "GRAPH_CONFLICT" for c in combined_claims)
    support_rate = supported / len(combined_claims) if combined_claims else None
    support_ok = support_rate is not None and support_rate >= threshold

    requirement_results = source_report["requirement_results"]
    requirements_satisfied = bool(requirement_results) and all(r["status"] == "SATISFIED" for r in requirement_results)

    valid = (
        bool(combined_claims)
        and support_ok
        and source_contradictions == 0
        and graph_conflicts == 0
        and not whole_contradicted
        and (requirements_satisfied if REQUIRE_ALL_QUESTION_PARTS else True)
    )

    evidence_refs = list(dict.fromkeys(
        sid for c in combined_claims
        for sid in (c["source_check"].get("evidence_source_ids", []) + c["symbolic_check"].get("evidence_source_ids", []))
    ))

    return {
        "valid": valid,
        "support_rate": round(support_rate, 3) if support_rate is not None else None,
        "total_claims": len(combined_claims),
        "supported_claims": supported,
        "source_contradictions": source_contradictions,
        "graph_conflicts": graph_conflicts,
        "whole_answer_verification": whole_answer,
        "requirements": requirements,
        "requirements_satisfied": requirements_satisfied,
        "requirement_results": requirement_results,
        "evidence_refs": evidence_refs,
        "all_claims": combined_claims,
        "reason": (
            f"support={support_rate}; threshold_ok={support_ok}; "
            f"contradictions={source_contradictions}; graph_conflicts={graph_conflicts}; "
            f"whole_answer={whole_answer['status']}; requirements_ok={requirements_satisfied}"
        ),
    }
