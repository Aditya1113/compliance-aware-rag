"""Evaluation Dashboard — benchmark results from the 50-question held-out test set."""

import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(APP_DIR, "data")

# --- Load data ---
@st.cache_data
def load_eval_data():
    retrieval_summary = pd.read_csv(os.path.join(DATA_DIR, "eval_retrieval_summary.csv"))
    retrieval_by_type = pd.read_csv(os.path.join(DATA_DIR, "eval_retrieval_by_type.csv"))
    cross_model = pd.read_csv(os.path.join(DATA_DIR, "eval_cross_model.csv"))

    with open(os.path.join(DATA_DIR, "eval_pipeline_results.json")) as f:
        pipeline_results = json.load(f)

    return retrieval_summary, retrieval_by_type, cross_model, pipeline_results


retrieval_summary, retrieval_by_type, cross_model, pipeline_results = load_eval_data()

# --- Header ---
st.title("Evaluation Dashboard")
st.markdown(
    "*Frozen benchmark results from a 50-question held-out test set across 5 question categories. "
    "All parameters were tuned on a separate 20-question DEV set and frozen before evaluation.*"
)

st.markdown("---")

# ============================================================
# Section 1: High-Level Metrics
# ============================================================
st.markdown("### Key Metrics")

total = len(pipeline_results)
accepted = sum(1 for r in pipeline_results if r["final_status"] == "ACCEPTED")
rejected = total - accepted
revised = sum(1 for r in pipeline_results if r["attempts"] > 1)
support_rates = [r["validation"]["support_rate"] for r in pipeline_results
                 if r.get("validation", {}).get("support_rate") is not None]
avg_support = sum(support_rates) / len(support_rates) if support_rates else 0
wall_times = [r["wall_time_s"] for r in pipeline_results if "wall_time_s" in r]
avg_time = sum(wall_times) / len(wall_times) if wall_times else 0

# Cross-model agreement
ext_routing = cross_model["external_routing"].str.strip()
pipe_status = cross_model["pipeline_status"].str.strip()
agree = ((ext_routing == "SHOULD_ACCEPT") & (pipe_status == "ACCEPTED")) | \
        ((ext_routing.isin(["SHOULD_WITHHOLD", "SHOULD_REJECT"])) & (pipe_status == "REJECTED"))
agreement_pct = agree.sum() / len(cross_model)

# External quality
ext_quality = cross_model["external_quality"].str.strip().value_counts()
correct = ext_quality.get("CORRECT", 0)
partial = ext_quality.get("PARTIALLY_CORRECT", 0)
incorrect = ext_quality.get("INCORRECT", 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accepted", f"{accepted}/{total}", f"{accepted/total*100:.0f}%")
c2.metric("Rejected", f"{rejected}/{total}", f"{rejected/total*100:.0f}%")
c3.metric("Revised", f"{revised}/{total}")
c4.metric("Avg Support Rate", f"{avg_support:.0%}")
c5.metric("Cross-Model Agreement", f"{agreement_pct:.0%}")
c6.metric("Avg Pipeline Time", f"{avg_time:.0f}s")

st.markdown("---")

# ============================================================
# Section 2: Pipeline Outcomes by Question Type
# ============================================================
st.markdown("### Pipeline Outcomes by Question Type")

type_labels = {
    "explicit_article": "Explicit Article",
    "uncited_semantic": "Uncited Semantic",
    "multi_part": "Multi-Part",
    "actor_scope": "Actor/Scope",
    "cross_provision_relational": "Cross-Provision",
}

type_data = []
for qt, label in type_labels.items():
    subset = [r for r in pipeline_results if r["question_type"] == qt]
    acc = sum(1 for r in subset if r["final_status"] == "ACCEPTED")
    type_data.append({
        "Question Type": label,
        "Total": len(subset),
        "Accepted": acc,
        "Rejected": len(subset) - acc,
        "Acceptance Rate": acc / len(subset) if subset else 0,
    })

type_df = pd.DataFrame(type_data)

col_chart, col_table = st.columns([3, 2])

with col_chart:
    chart_df = type_df[["Question Type", "Accepted", "Rejected"]].set_index("Question Type")
    st.bar_chart(chart_df)

with col_table:
    display_df = type_df.copy()
    display_df["Acceptance Rate"] = display_df["Acceptance Rate"].map("{:.0%}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================
# Section 3: Retrieval Performance
# ============================================================
st.markdown("### Retrieval Performance (Recall@K)")

k_select = st.radio("Select K", [3, 5, 10], index=1, horizontal=True)

ret_k = retrieval_summary[retrieval_summary["k"] == k_select].copy()
ret_k = ret_k.rename(columns={"system": "Strategy"})

col_recall, col_mrr = st.columns(2)

with col_recall:
    st.markdown(f"#### Recall@{k_select}")
    recall_chart = ret_k[["Strategy", "recall"]].set_index("Strategy")
    st.bar_chart(recall_chart)

with col_mrr:
    st.markdown(f"#### MRR@{k_select}")
    mrr_chart = ret_k[["Strategy", "mrr"]].set_index("Strategy")
    st.bar_chart(mrr_chart)

st.dataframe(
    ret_k[["Strategy", "hit", "recall", "precision", "mrr"]].round(3),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ============================================================
# Section 4: Retrieval by Question Type (at K=5)
# ============================================================
st.markdown("### Retrieval Recall@5 by Question Type")

ret_type = retrieval_by_type.copy()
ret_type["question_type"] = ret_type["question_type"].map(type_labels)
ret_type = ret_type.rename(columns={"system": "Strategy", "question_type": "Question Type"})

pivot = ret_type.pivot_table(
    index="Question Type", columns="Strategy", values="recall", aggfunc="first"
).round(2)

# Reorder columns
col_order = ["Dense Vector", "Legal-aware Vector", "Graph-only", "Legal-aware Hybrid"]
pivot = pivot[[c for c in col_order if c in pivot.columns]]

st.dataframe(pivot, use_container_width=True)
st.bar_chart(pivot)

st.caption(
    "Notice how **Legal-aware Vector** dominates on Explicit Article questions "
    "(Article resolver catches citations that dense similarity misses), while "
    "**Graph-only** excels on Cross-Provision questions where answers span multiple "
    "regulatory provisions connected by relationships."
)

st.markdown("---")

# ============================================================
# Section 5: Cross-Model Evaluation (GPT-4o vs Claude Sonnet)
# ============================================================
st.markdown("### Cross-Model Quality Assessment")
st.markdown(
    "*An independent Claude Sonnet evaluator reviewed all 50 answers against gold-standard "
    "references, without seeing the pipeline's own validation judgement.*"
)

col_qual, col_route = st.columns(2)

with col_qual:
    st.markdown("#### External Quality Rating")
    qual_df = pd.DataFrame({
        "Rating": ["Correct", "Partially Correct", "Incorrect"],
        "Count": [correct, partial, incorrect],
    }).set_index("Rating")
    st.bar_chart(qual_df)

with col_route:
    st.markdown("#### Routing Agreement")
    route_counts = cross_model.apply(
        lambda r: "Agree" if (
            (r["external_routing"].strip() == "SHOULD_ACCEPT" and r["pipeline_status"].strip() == "ACCEPTED") or
            (r["external_routing"].strip() in ("SHOULD_WITHHOLD", "SHOULD_REJECT") and r["pipeline_status"].strip() == "REJECTED")
        ) else "Disagree",
        axis=1
    ).value_counts()
    route_df = pd.DataFrame({"Count": route_counts}).reset_index()
    route_df.columns = ["Agreement", "Count"]
    st.bar_chart(route_df.set_index("Agreement"))

# Quality by question type
st.markdown("#### External Quality by Question Type")
cm = cross_model.copy()
cm["question_type"] = cm["question_type"].str.strip().map(type_labels)
cm["external_quality"] = cm["external_quality"].str.strip()

qual_pivot = pd.crosstab(cm["question_type"], cm["external_quality"])
qual_pivot = qual_pivot[[c for c in ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"] if c in qual_pivot.columns]]
st.dataframe(qual_pivot, use_container_width=True)

st.markdown("---")

# ============================================================
# Section 6: Per-Question Detail
# ============================================================
st.markdown("### Per-Question Results")
st.markdown("Explore individual question results from the benchmark.")

# Build combined table
detail_rows = []
for r in pipeline_results:
    cm_row = cross_model[cross_model["id"].str.strip() == r["id"]]
    ext_qual = cm_row["external_quality"].values[0].strip() if len(cm_row) > 0 else "N/A"
    ext_route = cm_row["external_routing"].values[0].strip() if len(cm_row) > 0 else "N/A"

    detail_rows.append({
        "ID": r["id"],
        "Type": type_labels.get(r["question_type"], r["question_type"]),
        "Status": r["final_status"],
        "Attempts": r["attempts"],
        "Support Rate": f"{r['validation']['support_rate']:.0%}" if r["validation"]["support_rate"] is not None else "N/A",
        "Ext. Quality": ext_qual,
        "Ext. Routing": ext_route,
        "Time (s)": f"{r['wall_time_s']:.0f}" if "wall_time_s" in r else "N/A",
    })

detail_df = pd.DataFrame(detail_rows)

# Filters
filter_cols = st.columns(3)
with filter_cols[0]:
    type_filter = st.multiselect("Question Type", list(type_labels.values()), default=list(type_labels.values()))
with filter_cols[1]:
    status_filter = st.multiselect("Pipeline Status", ["ACCEPTED", "REJECTED"], default=["ACCEPTED", "REJECTED"])
with filter_cols[2]:
    quality_filter = st.multiselect("External Quality", ["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"], default=["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"])

filtered = detail_df[
    detail_df["Type"].isin(type_filter) &
    detail_df["Status"].isin(status_filter) &
    detail_df["Ext. Quality"].isin(quality_filter)
]

st.dataframe(filtered, use_container_width=True, hide_index=True)

# Expandable detail for selected question
selected_id = st.selectbox("Select a question to see full detail", filtered["ID"].tolist() if not filtered.empty else [])

if selected_id:
    result = next((r for r in pipeline_results if r["id"] == selected_id), None)
    cm_row = cross_model[cross_model["id"].str.strip() == selected_id]

    if result:
        st.markdown(f"#### {selected_id}: {result['query']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Gold Answer:**")
            st.info(result.get("gold_answer", "N/A"))
        with col_b:
            st.markdown("**System Answer:**")
            if result["final_status"] == "ACCEPTED":
                st.success(result["answer"])
            else:
                st.warning(result["answer"])

        if len(cm_row) > 0:
            row = cm_row.iloc[0]
            st.markdown("**External Evaluator (Claude Sonnet):**")
            st.markdown(f"- Quality: **{row['external_quality'].strip()}**")
            st.markdown(f"- Routing: **{row['external_routing'].strip()}**")
            st.markdown(f"- Confidence: **{row['external_confidence']}**")
            if pd.notna(row.get("external_rationale")):
                with st.expander("Evaluator Rationale"):
                    st.markdown(row["external_rationale"])

st.markdown("---")
st.markdown("### Methodology")
st.markdown("""
- **Frozen benchmark**: 50 test questions across 5 categories, held out from parameter tuning
- **DEV/TEST split**: 20 DEV questions used for hyperparameter tuning (RRF weights, graph hops, relation exclusions), all frozen before TEST evaluation
- **Gold standard**: Each question has a human-written gold answer and gold source chunk IDs
- **Cross-model evaluation**: Independent Claude Sonnet evaluator assessed all answers without seeing the pipeline's own validation
- **Metrics**: Retrieval (Recall@K, MRR), Pipeline (acceptance rate, support rate), Quality (external CORRECT/PARTIAL/INCORRECT rating)
""")
