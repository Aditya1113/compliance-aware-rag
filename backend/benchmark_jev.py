"""Benchmark: Compare GPT-4o-mini vs Jev for claim verification and whole-answer validation.

Usage:
    export OPENAI_API_KEY=sk-...
    export TYPESAFE_API_KEY=ts-...
    cd app/backend
    python benchmark_jev.py

Outputs: jev_benchmark_results.json with per-task comparison.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    typesafe_key = os.environ.get("TYPESAFE_API_KEY")

    if not api_key:
        print("ERROR: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    if not typesafe_key:
        print("ERROR: Set TYPESAFE_API_KEY environment variable.")
        sys.exit(1)

    from typesafe_sdk import TypeSafeClient, Choice, Noul
    from langchain_openai import ChatOpenAI
    from core.config import SOURCE_VERIFIER_MODEL, CORPUS_PATH
    from core.corpus import load_corpus, format_doc_evidence, docs_from_source_ids
    from core.generation import invoke_with_metadata, parse_json_object

    # Load corpus
    print("Loading corpus...")
    corpus, corpus_by_id = load_corpus(CORPUS_PATH)

    # Load pipeline results (has pre-computed claims, answers, doc_ids)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    with open(os.path.join(data_dir, "eval_pipeline_results.json")) as f:
        pipeline_results = json.load(f)

    print(f"Loaded {len(pipeline_results)} pipeline results")

    # Setup clients
    gpt_llm = ChatOpenAI(model=SOURCE_VERIFIER_MODEL, temperature=0, api_key=api_key)
    jev_client = TypeSafeClient(api_key=typesafe_key)

    # Sample 10 questions for benchmark (mix of accepted/rejected)
    accepted = [r for r in pipeline_results if r["final_status"] == "ACCEPTED"][:7]
    rejected = [r for r in pipeline_results if r["final_status"] == "REJECTED"][:3]
    sample = accepted + rejected
    print(f"Benchmarking on {len(sample)} questions ({len(accepted)} accepted, {len(rejected)} rejected)")

    results = []

    for idx, result in enumerate(sample):
        print(f"\n--- [{idx+1}/{len(sample)}] {result['id']}: {result['query'][:80]}...")

        # Get docs
        doc_ids = [d["source_id"] for d in result.get("doc_evidence", [])]
        docs = docs_from_source_ids(doc_ids, corpus_by_id)
        doc_context = format_doc_evidence(docs)
        answer = result["answer"]
        query = result["query"]

        # =====================================================
        # Task 1: Whole-answer verification
        # =====================================================
        print("  [1] Whole-answer verification...")

        # GPT-4o-mini
        gpt_prompt = f"""You are a strict legal-grounding verifier for EU financial regulation.

QUESTION:
{query}

ANSWER TO VERIFY:
{answer}

AUTHORITATIVE REGULATORY EVIDENCE:
{doc_context}

Classify the WHOLE ANSWER as exactly one of:
SUPPORTED - The material claims are supported by the regulatory evidence and duties are attributed to the correct legal actor.
CONTRADICTED - At least one material claim conflicts with the evidence, or the answer attributes a duty to the wrong legal actor.
NOT_ENOUGH_EVIDENCE - The evidence is insufficient to determine correctness.

IMPORTANT: Legal actor alignment is mandatory.

Return ONLY valid JSON:
{{"status": "SUPPORTED|CONTRADICTED|NOT_ENOUGH_EVIDENCE", "reason": "A concise explanation."}}
"""
        gpt_start = time.perf_counter()
        gpt_call = invoke_with_metadata(gpt_llm, gpt_prompt)
        gpt_latency = time.perf_counter() - gpt_start
        gpt_parsed = parse_json_object(gpt_call["content"], {"status": "NOT_ENOUGH_EVIDENCE"})
        gpt_status = gpt_parsed.get("status", "NOT_ENOUGH_EVIDENCE").upper()
        print(f"    GPT: {gpt_status} ({gpt_latency:.2f}s)")

        # Jev
        jev_state = f"""QUESTION: {query}

ANSWER TO VERIFY: {answer}

REGULATORY EVIDENCE: {doc_context[:4000]}"""

        jev_start = time.perf_counter()
        try:
            jev_result = jev_client.system_one(
                jev_state,
                {
                    "verification": Choice(
                        instructions=(
                            "Is this answer supported by the regulatory evidence with correct legal actor attribution? "
                            "SUPPORTED means claims match evidence and duties are attributed to the correct legal actor. "
                            "CONTRADICTED means at least one claim conflicts with evidence or attributes a duty to the wrong actor. "
                            "NOT_ENOUGH_EVIDENCE means the evidence is insufficient."
                        ),
                        criteria={
                            "SUPPORTED": "Claims match evidence, correct legal actor attribution",
                            "CONTRADICTED": "Claims conflict with evidence or wrong legal actor",
                            "NOT_ENOUGH_EVIDENCE": "Evidence insufficient to determine",
                        },
                    ),
                    "has_contradiction": Noul(
                        instructions="Does the answer attribute any duty or responsibility to the wrong legal actor based on the evidence?"
                    ),
                },
            )
            jev_latency = time.perf_counter() - jev_start
            jev_status = jev_result.choices["verification"].choice
            jev_confidence = jev_result.choices["verification"].confidence
            jev_contradiction_prob = jev_result.nouls["has_contradiction"].noul
            print(f"    Jev: {jev_status} (conf={jev_confidence:.2f}, contradiction_prob={jev_contradiction_prob:.2f}, {jev_latency:.3f}s)")
        except Exception as e:
            jev_latency = time.perf_counter() - jev_start
            jev_status = "ERROR"
            jev_confidence = 0
            jev_contradiction_prob = 0
            print(f"    Jev ERROR: {e} ({jev_latency:.3f}s)")

        # =====================================================
        # Task 2: Per-claim verification (first 3 claims)
        # =====================================================
        claims = result.get("validation", {}).get("claims", [])[:3]
        claim_comparisons = []

        for ci, claim in enumerate(claims):
            claim_text = f"{claim['subject']} {claim['relation']} {claim['object']}"
            original_status = claim["final_status"]

            # Jev claim check
            claim_state = f"""CLAIM: {claim_text}
REGULATORY EVIDENCE: {doc_context[:3000]}
QUESTION CONTEXT: {query}"""

            jev_claim_start = time.perf_counter()
            try:
                jev_claim_result = jev_client.system_one(
                    claim_state,
                    {
                        "status": Choice(
                            instructions=(
                                "Is this specific claim supported by the regulatory evidence? "
                                "Check that the duty is attributed to the correct legal actor."
                            ),
                            criteria={
                                "SUPPORTED": "Claim is explicitly supported by evidence with correct actor",
                                "CONTRADICTED": "Claim conflicts with evidence or wrong actor",
                                "NOT_ENOUGH_EVIDENCE": "Evidence insufficient for this claim",
                            },
                        ),
                    },
                )
                jev_claim_latency = time.perf_counter() - jev_claim_start
                jev_claim_status = jev_claim_result.choices["status"].choice
                jev_claim_conf = jev_claim_result.choices["status"].confidence
            except Exception as e:
                jev_claim_latency = time.perf_counter() - jev_claim_start
                jev_claim_status = "ERROR"
                jev_claim_conf = 0

            claim_comparisons.append({
                "claim": claim_text,
                "pipeline_status": original_status,
                "jev_status": jev_claim_status,
                "jev_confidence": round(jev_claim_conf, 3),
                "jev_latency_ms": round(jev_claim_latency * 1000, 1),
            })

        if claim_comparisons:
            print(f"  [2] Claim verification ({len(claim_comparisons)} claims):")
            for cc in claim_comparisons:
                print(f"    Pipeline={cc['pipeline_status']}, Jev={cc['jev_status']} (conf={cc['jev_confidence']}, {cc['jev_latency_ms']}ms)")

        results.append({
            "id": result["id"],
            "question_type": result["question_type"],
            "query": query[:100],
            "pipeline_final_status": result["final_status"],
            "whole_answer": {
                "gpt_status": gpt_status,
                "gpt_latency_s": round(gpt_latency, 3),
                "jev_status": jev_status,
                "jev_confidence": round(jev_confidence, 3) if jev_status != "ERROR" else None,
                "jev_contradiction_prob": round(jev_contradiction_prob, 3) if jev_status != "ERROR" else None,
                "jev_latency_s": round(jev_latency, 3),
                "speedup": round(gpt_latency / jev_latency, 1) if jev_latency > 0 else None,
                "agree": gpt_status == jev_status,
            },
            "claim_comparisons": claim_comparisons,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    whole_agree = sum(1 for r in results if r["whole_answer"]["agree"])
    total = len(results)
    avg_gpt_latency = sum(r["whole_answer"]["gpt_latency_s"] for r in results) / total
    avg_jev_latency = sum(r["whole_answer"]["jev_latency_s"] for r in results) / total
    avg_speedup = avg_gpt_latency / avg_jev_latency if avg_jev_latency > 0 else 0

    all_claims = [cc for r in results for cc in r["claim_comparisons"]]
    claim_agree = sum(1 for cc in all_claims if _statuses_agree(cc["pipeline_status"], cc["jev_status"]))

    print(f"Whole-answer agreement (GPT vs Jev): {whole_agree}/{total} ({whole_agree/total*100:.0f}%)")
    print(f"Claim-level agreement (Pipeline vs Jev): {claim_agree}/{len(all_claims)} ({claim_agree/len(all_claims)*100:.0f}%)" if all_claims else "No claims")
    print(f"Avg GPT latency: {avg_gpt_latency:.2f}s")
    print(f"Avg Jev latency: {avg_jev_latency:.3f}s")
    print(f"Avg speedup: {avg_speedup:.0f}x")

    summary = {
        "total_questions": total,
        "whole_answer_agreement": f"{whole_agree}/{total}",
        "whole_answer_agreement_pct": round(whole_agree / total * 100),
        "claim_level_agreement": f"{claim_agree}/{len(all_claims)}" if all_claims else "N/A",
        "claim_level_agreement_pct": round(claim_agree / len(all_claims) * 100) if all_claims else None,
        "avg_gpt_latency_s": round(avg_gpt_latency, 3),
        "avg_jev_latency_s": round(avg_jev_latency, 3),
        "avg_speedup_x": round(avg_speedup, 1),
        "results": results,
    }

    out_path = os.path.join(data_dir, "jev_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _statuses_agree(pipeline_status, jev_status):
    """Check if pipeline claim status and Jev status agree."""
    supported = {"SUPPORTED_SOURCE", "SUPPORTED_GRAPH", "SUPPORTED"}
    contradicted = {"CONTRADICTED_SOURCE", "GRAPH_CONFLICT", "CONTRADICTED"}
    if pipeline_status in supported and jev_status in supported:
        return True
    if pipeline_status in contradicted and jev_status in contradicted:
        return True
    if pipeline_status == "UNVERIFIED" and jev_status == "NOT_ENOUGH_EVIDENCE":
        return True
    return False


if __name__ == "__main__":
    main()
