"""Answer generation and revision prompts."""

import time
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .config import GENERATION_MODEL, RELATION_TYPES
from .corpus import format_doc_evidence
from .graph import format_graph_evidence


def create_llm(api_key, model=GENERATION_MODEL):
    return ChatOpenAI(model=model, temperature=0, api_key=api_key)


def invoke_with_metadata(model, prompt_text):
    start = time.perf_counter()
    response = model.invoke(prompt_text)
    elapsed = time.perf_counter() - start
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = (getattr(response, "response_metadata", {}) or {}).get("token_usage", {})
    return {
        "content": response.content.strip(),
        "latency_s": round(elapsed, 3),
        "token_usage": usage or {},
    }


def parse_json_object(raw_text, default=None):
    default = {} if default is None else default
    raw = re.sub(r"```json|```", "", (raw_text or "")).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return default
        try:
            return json.loads(match.group())
        except Exception:
            return default


def is_answered(answer):
    text = (answer or "").strip().lower().rstrip(".")
    return bool(text) and text != "not found in context"


PIPELINE_DRAFT_PROMPT = PromptTemplate.from_template("""You are answering a question about EU financial regulation.

AUTHORITATIVE DOCUMENT EVIDENCE:
{doc_context}

SECONDARY GRAPH EVIDENCE:
{graph_context}

QUESTION:
{question}

EXPLICIT ANSWER REQUIREMENTS:
{requirements}

RULES:

1. The regulatory document evidence is authoritative.
2. Graph evidence may help with relationships but must never override the regulatory document evidence.
3. Answer every listed requirement explicitly.
4. Use no outside knowledge.
5. Ensure that each duty, responsibility, condition, deadline, recipient, permission or prohibition is attributed to the correct legal actor, function and provision.
6. A statement may appear somewhere in the evidence but still be incorrect for the question if it belongs to a different legal actor, function, paragraph or obligation.
7. Include only information that directly answers the question and is explicitly supported by the regulatory document evidence.
8. Do not add generic conclusions, implications, benefits, purposes, interpretations, summaries or explanatory closing statements unless they are explicitly stated in the regulatory evidence.
9. Do not infer consequences merely because they appear reasonable.
10. Prefer a concise regulatory answer containing only the duties, responsibilities, conditions, deadlines, recipients, permissions, prohibitions or facts necessary to answer the question.
11. End the answer after the final evidence-supported point. Do not append a generic explanatory conclusion.
12. If the evidence does not support a particular requirement, state that limitation rather than inventing an answer.
13. If the evidence is wholly insufficient, say exactly: "Not found in context."

LEGAL SCOPE SELECTION:

14. Before answering, identify the specific legal actor or function asked about in the QUESTION.
15. Prefer regulatory evidence that explicitly assigns duties or responsibilities to that actor or function.
16. Do NOT combine responsibility lists with general obligations imposed on the investment firm, conditions for independence, proportionality exceptions, or neighbouring paragraph duties.
17. Different paragraphs within the same Article may apply to different legal actors. Do not treat them as interchangeable.

Answer:
""")


REVISION_PROMPT_TEMPLATE = """You are revising an answer about EU financial regulation.

AUTHORITATIVE REGULATORY DOCUMENT EVIDENCE:
{doc_context}

SECONDARY GRAPH EVIDENCE:
{graph_context}

QUESTION:
{question}

EXPLICIT QUESTION REQUIREMENTS:
{requirements}

PREVIOUS ANSWER:
{previous_answer}

CLAIM-LEVEL FEEDBACK:
{claim_feedback}

QUESTION-COMPLETENESS FEEDBACK:
{requirement_feedback}

WHOLE-ANSWER LEGAL VERIFICATION:
{whole_answer_feedback}

REVISION RULES:
1. The regulatory DOCUMENT evidence is authoritative.
2. Correct every issue identified by the whole-answer verifier.
3. Pay particular attention to LEGAL ATTRIBUTION.
4. If the whole-answer status is CONTRADICTED, reconstruct the answer from the authoritative regulatory text.
5. Answer every explicit question requirement.
6. Do not use outside knowledge.
7. Graph evidence must never override document evidence.
8. Remove, correct or qualify anything that is not properly supported.
9. If the evidence is wholly insufficient, say exactly: "Not found in context."
10. Remove unsupported explanatory or concluding statements.
11. End the revised answer after the final evidence-supported point.

REVISED ANSWER:
"""


def render_requirements(requirements):
    return "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))


def generate_draft(llm, docs, graph_evidence, question, requirements):
    call = invoke_with_metadata(
        llm,
        PIPELINE_DRAFT_PROMPT.format(
            doc_context=format_doc_evidence(docs),
            graph_context=format_graph_evidence(graph_evidence),
            question=question,
            requirements=render_requirements(requirements),
        ),
    )
    return call["content"], call["latency_s"]


def generate_revision(llm, docs, graph_evidence, question, requirements,
                      previous_answer, validation):
    failed_claims = [
        c for c in validation.get("all_claims", [])
        if c["final_status"] not in {"SUPPORTED_SOURCE", "SUPPORTED_GRAPH"}
    ]
    failed_requirements = [
        item for item in validation.get("requirement_results", [])
        if item["status"] != "SATISFIED"
    ]

    claim_feedback = "\n".join(
        f"- {c['subject']} / {c['relation']} / {c['object']} -> {c['final_status']}"
        for c in failed_claims
    ) or "No atomic claim was individually rejected."

    requirement_feedback = "\n".join(
        f"- Requirement {item['requirement_index'] + 1}: {item['requirement']} -> {item['status']} ({item['reason']})"
        for item in failed_requirements
    ) or "All explicit question requirements were satisfied."

    whole_check = validation.get("whole_answer_verification", {})
    whole_answer_feedback = (
        f"Status: {whole_check.get('status', 'NOT_ENOUGH_EVIDENCE')}\n"
        f"Reason: {whole_check.get('reason', '')}"
    )

    prompt = REVISION_PROMPT_TEMPLATE.format(
        doc_context=format_doc_evidence(docs),
        graph_context=format_graph_evidence(graph_evidence),
        question=question,
        requirements=render_requirements(requirements),
        previous_answer=previous_answer,
        claim_feedback=claim_feedback,
        requirement_feedback=requirement_feedback,
        whole_answer_feedback=whole_answer_feedback,
    )

    call = invoke_with_metadata(llm, prompt)
    return call["content"], call["latency_s"]
