"""
app.py
------
Streamlit application for the Human Relevance Judgment System.

Supports two datasets:
  • Synthetic Demo  — 40 hand-crafted documents (documents.csv)
  • SciFact         — BEIR SciFact corpus (documents_scifact.csv)
                      Run  python scifact_import.py  first to generate it.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from retrieval import RetrievalEngine
from judgment_manager import (
    save_judgments,
    load_judgments_for_query,
    load_all_judgments,
    list_judged_queries,
    get_relevance_dict,
    get_ranked_doc_ids,
    delete_query,
    label_for,
    LABEL_MAP,
)
from evaluation import (
    evaluate_query,
    mean_average_precision,
    average_precision,
)
from benchmark import (
    scifact_available,
    load_queries as load_scifact_queries,
    get_qrels_for_query,
    count_relevant_in_qrels,
    benchmark_evaluate,
    multi_query_map,
)


# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IR Human Relevance Judgment System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px; padding: 20px 16px; text-align: center;
        color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.25); margin-bottom: 8px;
    }
    .metric-card .metric-label  { font-size:.85rem; opacity:.85; text-transform:uppercase;
                                   letter-spacing:.05em; margin-bottom:6px; }
    .metric-card .metric-value  { font-size:2.2rem; font-weight:700; }
    .metric-card .metric-sub    { font-size:.78rem; opacity:.7; margin-top:4px; }
    .doc-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-left:4px solid #2d6a9f; border-radius:8px;
        padding:16px; margin-bottom:12px;
    }
    .doc-title  { font-size:1.05rem; font-weight:600; color:#1e3a5f; }
    .doc-meta   { font-size:.8rem; color:#718096; margin:4px 0 8px 0; }
    .doc-snippet{ font-size:.9rem; color:#4a5568; line-height:1.6; }
    .section-header {
        font-size:1.3rem; font-weight:700; color:#1e3a5f;
        border-bottom:2px solid #2d6a9f; padding-bottom:6px; margin:20px 0 14px 0;
    }
    .bench-note {
        background:#fffbea; border:1px solid #f6ad55;
        border-radius:8px; padding:12px 16px; font-size:.88rem; color:#744210;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset configuration
# ──────────────────────────────────────────────────────────────────────────────

DATASET_OPTIONS = ["Synthetic Demo"]
if scifact_available():
    DATASET_OPTIONS.append("SciFact")

DATASET_CSV = {
    "Synthetic Demo": "documents.csv",
    "SciFact":        "documents_scifact.csv",
}

DATASET_KEY = {          # used to scope judgments
    "Synthetic Demo": "synthetic",
    "SciFact":        "scifact",
}


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "search_results":    pd.DataFrame(),
        "current_query":     "",
        "judgments":         {},
        "saved":             False,
        "top_k":             10,
        "active_dataset":    "Synthetic Demo",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Cached engine — rebuilds ONLY when the CSV path changes
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building TF-IDF index…")
def _build_engine(csv_path: str) -> RetrievalEngine:
    return RetrievalEngine(csv_path)


def get_engine(dataset_name: str) -> RetrievalEngine:
    csv_path = os.path.abspath(DATASET_CSV[dataset_name])
    return _build_engine(csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/search--v1.png", width=60)
    st.title("IR Judgment System")
    st.markdown("---")

    # ── Dataset selector ──────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    selected_dataset = st.selectbox(
        "Active dataset",
        DATASET_OPTIONS,
        index=DATASET_OPTIONS.index(st.session_state["active_dataset"])
              if st.session_state["active_dataset"] in DATASET_OPTIONS else 0,
        key="sidebar_dataset",
    )

    # When the user switches datasets, clear the search state to avoid
    # showing stale results from the other collection.
    if selected_dataset != st.session_state["active_dataset"]:
        st.session_state["active_dataset"]  = selected_dataset
        st.session_state["search_results"]  = pd.DataFrame()
        st.session_state["current_query"]   = ""
        st.session_state["judgments"]       = {}
        st.session_state["saved"]           = False
        st.rerun()

    active_ds  = st.session_state["active_dataset"]
    ds_key     = DATASET_KEY[active_ds]
    engine     = get_engine(active_ds)

    if not scifact_available() and "SciFact" not in DATASET_OPTIONS:
        st.info("SciFact not imported yet.\nRun: `python scifact_import.py`")

    st.markdown("---")
    st.markdown("### ⚙️ Search Settings")
    top_k = st.slider("Documents to retrieve (k)", 5, 20, 10)
    st.session_state["top_k"] = top_k

    st.markdown("---")
    st.markdown("### 📚 Collection Info")
    total_docs = len(engine.get_all_documents())
    st.metric("Documents", total_docs)
    st.metric("Active dataset", active_ds)

    st.markdown("---")
    st.markdown("### 📋 Judged Queries")
    judged_df = list_judged_queries(dataset=ds_key)
    if judged_df.empty:
        st.info("No queries judged yet.")
    else:
        for _, row in judged_df.iterrows():
            label = row["query_text"]
            st.markdown(
                f"**{label[:35]}{'…' if len(label) > 35 else ''}**  \n"
                f"<small>{row['num_judgments']} docs · {row['timestamp'][:10]}</small>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### 📖 Grade Legend")
    for g, lbl in sorted(LABEL_MAP.items(), reverse=True):
        icons = {3: "🟢", 2: "🔵", 1: "🟡", 0: "🔴"}
        st.markdown(f"{icons[g]} **{g}** — {lbl}")


# ──────────────────────────────────────────────────────────────────────────────
# Tab layout — add Benchmark tab only when SciFact is available
# ──────────────────────────────────────────────────────────────────────────────

tab_labels = ["🔍 Search & Judge", "📊 Dashboard", "🗂️ History", "ℹ️ About"]
if scifact_available():
    tab_labels.insert(3, "🏆 Benchmark")

tabs = st.tabs(tab_labels)

# Map tab positions to names
_tab_map = {name: tabs[i] for i, name in enumerate(tab_labels)}


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Search & Judge
# ════════════════════════════════════════════════════════════════════════════════

with _tab_map["🔍 Search & Judge"]:
    st.markdown(
        f'<div class="section-header">Step 1 · Enter Your Query '
        f'<span style="font-size:.9rem;color:#718096;">({active_ds})</span></div>',
        unsafe_allow_html=True,
    )

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query_input = st.text_input(
            "Search query",
            placeholder="e.g. Applications of machine learning in healthcare",
            label_visibility="collapsed",
        )
    with col_btn:
        search_clicked = st.button("🔍 Search", use_container_width=True, type="primary")

    # Example queries change per dataset
    if active_ds == "Synthetic Demo":
        example_queries = [
            "machine learning healthcare",
            "deep learning medical imaging",
            "natural language processing",
            "drug discovery AI",
        ]
    else:
        example_queries = [
            "COVID-19 vaccine efficacy",
            "cancer immunotherapy",
            "CRISPR gene editing",
            "Alzheimer disease treatment",
        ]

    st.markdown("**Try these example queries:**")
    ex_cols = st.columns(4)
    for i, eq in enumerate(example_queries):
        if ex_cols[i].button(eq, key=f"ex_{i}"):
            query_input    = eq
            search_clicked = True

    # ── Run search ────────────────────────────────────────────────────────────

    if search_clicked and query_input.strip():
        results = engine.search(query_input.strip(), top_k=st.session_state["top_k"])
        st.session_state["search_results"] = results
        st.session_state["current_query"]  = query_input.strip()
        st.session_state["saved"]          = False

        prior = get_relevance_dict(query_input.strip(), dataset=ds_key)
        st.session_state["judgments"] = {
            row["doc_id"]: {
                "judgment": prior.get(row["doc_id"], 0),
                "title":    row["title"],
                "rank":     row["rank"],
            }
            for _, row in results.iterrows()
        }
    elif search_clicked and not query_input.strip():
        st.warning("Please enter a query before searching.")

    # ── Display results ───────────────────────────────────────────────────────

    results_df    = st.session_state["search_results"]
    current_query = st.session_state["current_query"]

    if not results_df.empty:
        st.markdown(
            f'<div class="section-header">Step 2 · Retrieved Documents for: '
            f'<em>"{current_query}"</em></div>',
            unsafe_allow_html=True,
        )
        st.info(
            f"📄 Showing top {len(results_df)} documents ranked by TF-IDF cosine similarity. "
            "Rate each document using the relevance selector."
        )

        # Check if the docs CSV has a 'url' column (SciFact does)
        all_docs    = engine.get_all_documents()
        has_url_col = "url" in all_docs.columns

        for _, row in results_df.iterrows():
            doc_id        = row["doc_id"]
            current_grade = st.session_state["judgments"].get(doc_id, {}).get("judgment", 0)

            # Get URL if available
            url_html = ""
            if has_url_col:
                url_rows = all_docs[all_docs["doc_id"] == doc_id]
                if not url_rows.empty:
                    url_val = str(url_rows.iloc[0].get("url", "")).strip()
                    if url_val and url_val not in ("nan", ""):
                        url_html = (
                            f'<a href="{url_val}" target="_blank" '
                            f'style="font-size:.78rem;color:#2b6cb0;">🔗 Source</a> &nbsp;|&nbsp; '
                        )

            with st.container():
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div class="doc-title">#{row['rank']} · {row['title']}</div>
                        <div class="doc-meta">
                            {url_html}📄 {doc_id} &nbsp;|&nbsp;
                            Similarity: <strong>{row['similarity_score']:.4f}</strong>
                        </div>
                        <div class="doc-snippet">{row['snippet']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                grade_options = [
                    "3 — Highly Relevant",
                    "2 — Relevant",
                    "1 — Partially Relevant",
                    "0 — Not Relevant",
                ]
                option_map = {3: 0, 2: 1, 1: 2, 0: 3}
                selected = st.selectbox(
                    "Your relevance judgment",
                    options=grade_options,
                    index=option_map.get(current_grade, 3),
                    key=f"sel_{doc_id}",
                    label_visibility="collapsed",
                )
                grade_val = int(selected[0])
                st.session_state["judgments"][doc_id] = {
                    "judgment": grade_val,
                    "title":    row["title"],
                    "rank":     row["rank"],
                }
                st.divider()

        # ── Save ──────────────────────────────────────────────────────────────

        st.markdown('<div class="section-header">Step 3 · Save Your Judgments</div>',
                    unsafe_allow_html=True)

        col_save, col_clear = st.columns([2, 1])
        with col_save:
            if st.button("💾 Save Judgments & Compute Metrics", type="primary",
                         use_container_width=True):
                qid = save_judgments(
                    current_query,
                    st.session_state["judgments"],
                    dataset=ds_key,
                )
                st.session_state["saved"] = True
                st.success(
                    f"✅ Judgments saved for **\"{current_query}\"** "
                    f"in **{active_ds}** (ID: {qid}). "
                    "Go to the **Dashboard** tab to see evaluation metrics!"
                )
                st.balloons()

        with col_clear:
            if st.button("🗑️ Clear Ratings", use_container_width=True):
                for did in st.session_state["judgments"]:
                    st.session_state["judgments"][did]["judgment"] = 0
                st.rerun()

    else:
        st.markdown("---")
        st.markdown("""
### How to use this system

1. **Choose a dataset** in the sidebar (Synthetic Demo or SciFact).
2. **Enter a query** and click Search.
3. **Rate each retrieved document** using the four-point relevance scale.
4. **Save your judgments** — they are scoped to the active dataset.
5. **Visit the Dashboard tab** to see Precision, Recall, MAP and NDCG.
6. **Visit the Benchmark tab** (SciFact only) to evaluate against official qrels.
""")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dashboard
# ════════════════════════════════════════════════════════════════════════════════

with _tab_map["📊 Dashboard"]:
    st.markdown(
        f'<div class="section-header">Evaluation Dashboard — {active_ds}</div>',
        unsafe_allow_html=True,
    )

    judged_queries = list_judged_queries(dataset=ds_key)

    if judged_queries.empty:
        st.info("No judgments saved yet for this dataset. Go to **Search & Judge** first.")
    else:
        query_list     = judged_queries["query_text"].tolist()
        selected_query = st.selectbox("Select a query to evaluate:", query_list,
                                      key="dash_query")
        k_dash = st.slider("Cut-off k", 1, 20, 10, key="dash_k")

        rel_dict   = get_relevance_dict(selected_query, dataset=ds_key)
        ranked_ids = get_ranked_doc_ids(selected_query, dataset=ds_key)

        if not rel_dict:
            st.warning("No judgments found for this query.")
        else:
            metrics = evaluate_query(ranked_ids, rel_dict, k=k_dash)

            # Metric cards
            st.markdown("### 📐 Evaluation Metrics")
            m1, m2, m3, m4 = st.columns(4)
            card_data = [
                (m1, "Precision@k", metrics["precision"],
                 "Fraction of top-k docs that are relevant"),
                (m2, "Recall@k",    metrics["recall"],
                 "Fraction of all relevant docs retrieved"),
                (m3, "AP (→MAP)",   metrics["ap"],
                 "Average Precision for this query"),
                (m4, "NDCG@k",      metrics["ndcg"],
                 "Graded ranking quality (0–1)"),
            ]
            for col, lbl, val, sub in card_data:
                col.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value">{val:.3f}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # MAP across all queries
            st.markdown("### 🗺️ MAP Across All Judged Queries")
            all_q_data = []
            for _, qrow in judged_queries.iterrows():
                qt = qrow["query_text"]
                rd = get_relevance_dict(qt, dataset=ds_key)
                ri = get_ranked_doc_ids(qt, dataset=ds_key)
                if rd and ri:
                    ap_val = average_precision(ri, rd, k=k_dash)
                    all_q_data.append({
                        "ranked_doc_ids": ri,
                        "relevance_dict": rd,
                        "k": k_dash,
                        "_ap": ap_val,
                        "query_text": qt,
                    })

            if all_q_data:
                map_val = mean_average_precision(all_q_data)
                st.metric("MAP across all queries", f"{map_val:.4f}")

                fig_map, ax_map = plt.subplots(figsize=(7, max(3, len(all_q_data) * 0.6)))
                q_labels = [d["query_text"][:30] + ("…" if len(d["query_text"]) > 30 else "")
                            for d in all_q_data]
                ap_vals  = [d["_ap"] for d in all_q_data]
                bars = ax_map.barh(q_labels, ap_vals, color="#2d6a9f", edgecolor="white")
                ax_map.set_xlim(0, 1.05)
                ax_map.set_xlabel("Average Precision")
                ax_map.set_title("AP per Query")
                ax_map.axvline(map_val, color="tomato", linestyle="--",
                               linewidth=1.5, label=f"MAP = {map_val:.3f}")
                ax_map.legend(fontsize=8)
                ax_map.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
                fig_map.tight_layout()
                st.pyplot(fig_map)
                plt.close(fig_map)

            # Grade distribution
            st.markdown("### 🎨 Relevance Grade Distribution")
            jdf = load_judgments_for_query(selected_query, dataset=ds_key)
            grade_counts = (
                jdf["judgment"].value_counts().reindex([3, 2, 1, 0], fill_value=0)
            )
            colours_bar = ["#276749", "#2b6cb0", "#d69e2e", "#c53030"]
            labels_bar  = ["Highly Relevant (3)", "Relevant (2)",
                           "Partially Relevant (1)", "Not Relevant (0)"]
            fig_d, ax_d = plt.subplots(figsize=(6, 3))
            bd = ax_d.bar(labels_bar, grade_counts.values,
                          color=colours_bar, edgecolor="white", width=0.5)
            ax_d.set_ylabel("Number of Documents")
            ax_d.set_title(f'Grade Distribution — "{selected_query[:40]}"')
            ax_d.bar_label(bd, padding=3, fontsize=9)
            ax_d.set_ylim(0, max(grade_counts.max() + 2, 5))
            fig_d.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)

            # Ranked table
            st.markdown("### 📋 Ranked Document Table")
            table_data = []
            for rank_i, did in enumerate(ranked_ids[:k_dash], start=1):
                grade    = rel_dict.get(did, 0)
                jdf_row  = jdf[jdf["doc_id"] == did]
                title    = jdf_row["title"].values[0] if not jdf_row.empty else did
                table_data.append({
                    "Rank":           rank_i,
                    "Doc ID":         did,
                    "Title":          title,
                    "Grade":          grade,
                    "Relevance Label":label_for(grade),
                    "Binary Relevant":"✅" if grade >= 2 else "❌",
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True,
                         hide_index=True)

            # Export
            st.markdown("### 💾 Export")
            export_df = jdf.copy()
            export_df["relevance_label"] = export_df["judgment"].apply(label_for)
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download judgments as CSV",
                data=csv_bytes,
                file_name=f"judgments_{ds_key}_{selected_query[:30].replace(' ','_')}.csv",
                mime="text/csv",
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Benchmark  (SciFact only)
# ════════════════════════════════════════════════════════════════════════════════

if "🏆 Benchmark" in _tab_map:
    with _tab_map["🏆 Benchmark"]:
        st.markdown(
            '<div class="section-header">Benchmark Evaluation — SciFact Official Qrels</div>',
            unsafe_allow_html=True,
        )
        st.markdown("""
        This tab evaluates our **TF-IDF retrieval engine** against the **official SciFact
        relevance judgments** (qrels).

        - Relevance labels here are the **official binary annotations** (0 = not relevant,
          1 = relevant) from the BEIR SciFact dataset.
        - They are **completely separate** from the human judgments (0–3) you enter in
          the Search & Judge tab.
        - Unjudged documents (not in qrels for a query) are treated as non-relevant
          following the **standard pooling assumption**. See the recall note for each query.
        """)

        # Load SciFact queries
        try:
            sf_queries = load_scifact_queries()
        except FileNotFoundError:
            st.error("scifact_queries.csv not found. Run `python scifact_import.py` first.")
            st.stop()

        # Query selector
        query_options = sf_queries["query_text"].tolist()
        query_ids     = sf_queries["query_id"].tolist()
        bench_q_text  = st.selectbox(
            "Select a benchmark query:", query_options, key="bench_query"
        )
        bench_q_idx = query_options.index(bench_q_text)
        bench_q_id  = query_ids[bench_q_idx]

        k_bench = st.slider("Cut-off k for benchmark metrics", 1, 20, 10, key="bench_k")

        # Retrieve with our TF-IDF engine (SciFact dataset must be active)
        if active_ds != "SciFact":
            st.warning(
                "Switch the dataset to **SciFact** in the sidebar to search "
                "the SciFact corpus for benchmark evaluation."
            )
        else:
            if st.button("🔎 Retrieve & Evaluate", type="primary"):
                # Retrieve
                bench_results = engine.search(bench_q_text, top_k=k_bench)
                if bench_results.empty:
                    st.warning("No results returned. Try a different query.")
                else:
                    ranked_ids_bench = bench_results["doc_id"].astype(str).tolist()

                    # Load official qrels for this query
                    qrels_dict = get_qrels_for_query(bench_q_id)
                    n_relevant  = count_relevant_in_qrels(bench_q_id)

                    if not qrels_dict:
                        st.warning(
                            f"No official qrels found for query ID {bench_q_id}. "
                            "This query may not have any annotated documents."
                        )
                    else:
                        # Evaluate
                        bench_metrics = benchmark_evaluate(
                            ranked_ids_bench, qrels_dict, k=k_bench
                        )

                        # Recall caveat banner
                        st.markdown(
                            f'<div class="bench-note">⚠️ <strong>Recall note:</strong> '
                            f'{bench_metrics["recall_note"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("")

                        # Metric cards
                        st.markdown("### 📐 Benchmark Metrics")
                        bm1, bm2, bm3, bm4 = st.columns(4)
                        bench_cards = [
                            (bm1, "Precision@k", bench_metrics["precision"],
                             f"Relevant in top-{k_bench}"),
                            (bm2, "Recall@k",    bench_metrics["recall"],
                             f"Of {n_relevant} judged-relevant docs"),
                            (bm3, "AP",          bench_metrics["ap"],
                             "Average Precision"),
                            (bm4, "NDCG@k",      bench_metrics["ndcg"],
                             "Ranking quality (binary)"),
                        ]
                        for col, lbl, val, sub in bench_cards:
                            col.markdown(
                                f"""<div class="metric-card">
                                    <div class="metric-label">{lbl}</div>
                                    <div class="metric-value">{val:.3f}</div>
                                    <div class="metric-sub">{sub}</div>
                                </div>""",
                                unsafe_allow_html=True,
                            )

                        # Stats row
                        st.markdown("### 📊 Qrel Coverage")
                        cs1, cs2, cs3 = st.columns(3)
                        cs1.metric("Judged-relevant in qrels", bench_metrics["relevant_in_qrels"])
                        cs2.metric(f"Top-{k_bench} docs with a qrel entry",
                                   bench_metrics["judged_in_top_k"])
                        cs3.metric(f"Unjudged in top-{k_bench}",
                                   bench_metrics["unjudged_in_top_k"])

                        # Retrieved documents table
                        st.markdown("### 📋 Retrieved Documents vs Qrels")
                        table_rows = []
                        for _, rrow in bench_results.iterrows():
                            did        = str(rrow["doc_id"])
                            qrel_grade = qrels_dict.get(did, None)
                            table_rows.append({
                                "Rank":              rrow["rank"],
                                "Doc ID":            did,
                                "Title":             rrow["title"],
                                "Similarity":        rrow["similarity_score"],
                                "Official Qrel":     qrel_grade if qrel_grade is not None else "—",
                                "Relevant (qrel)":   "✅" if qrel_grade == 1 else
                                                     ("❌" if qrel_grade == 0 else "⬜ Unjudged"),
                            })
                        st.dataframe(
                            pd.DataFrame(table_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Multi-query MAP note
                        st.markdown("---")
                        st.info(
                            "To compute MAP across multiple benchmark queries, "
                            "retrieve and evaluate each query individually. "
                            "MAP = average of AP values across queries."
                        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB — History
# ════════════════════════════════════════════════════════════════════════════════

with _tab_map["🗂️ History"]:
    st.markdown(
        f'<div class="section-header">Judgment History — {active_ds}</div>',
        unsafe_allow_html=True,
    )

    all_jdf = load_all_judgments()
    ds_jdf  = all_jdf[all_jdf["dataset"] == ds_key] if not all_jdf.empty else all_jdf

    if ds_jdf.empty:
        st.info(f"No judgments saved yet for **{active_ds}**.")
    else:
        judged_q = list_judged_queries(dataset=ds_key)
        st.success(
            f"**{len(judged_q)} queries** judged · "
            f"**{len(ds_jdf)} total judgments** in {active_ds}"
        )

        st.markdown("#### Query Summary")
        st.dataframe(
            judged_q.rename(columns={
                "query_id":     "Query ID",
                "query_text":   "Query",
                "num_judgments":"# Docs Judged",
                "timestamp":    "Last Updated",
            }),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### All Judgments")
        disp = ds_jdf.copy()
        disp["relevance_label"] = disp["judgment"].apply(label_for)
        st.dataframe(
            disp[["dataset","query_id","query_text","rank","doc_id",
                  "title","judgment","relevance_label","timestamp"]],
            use_container_width=True, hide_index=True,
        )

        csv_all = ds_jdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇️ Download all {active_ds} judgments as CSV",
            data=csv_all,
            file_name=f"all_judgments_{ds_key}.csv",
            mime="text/csv",
        )

        st.markdown("#### Delete Query Judgments")
        if not judged_q.empty:
            q_to_del = st.selectbox(
                "Select query to delete:", judged_q["query_text"].tolist(), key="del_q"
            )
            if st.button("🗑️ Delete selected query's judgments", type="secondary"):
                if delete_query(q_to_del, dataset=ds_key):
                    st.success(f"Deleted judgments for: **{q_to_del}**")
                    st.rerun()
                else:
                    st.error("Could not find judgments to delete.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB — About
# ════════════════════════════════════════════════════════════════════════════════

with _tab_map["ℹ️ About"]:
    st.markdown('<div class="section-header">About This System</div>',
                unsafe_allow_html=True)
    st.markdown("""
## 🔍 Human Relevance Judgment System

End-to-end IR evaluation pipeline supporting two datasets.

| Component          | Technology                              |
|--------------------|------------------------------------------|
| Document retrieval | TF-IDF + Cosine Similarity (scikit-learn) |
| User interface     | Streamlit                                |
| Judgment storage   | CSV (dataset-scoped)                     |
| Evaluation metrics | Custom Python                            |
| Benchmark qrels    | BEIR SciFact (official binary labels)    |

---

## 📂 Datasets

**Synthetic Demo** — 40 hand-crafted documents covering healthcare AI, NLP,
climate change, education technology, and more. Use this for learning the system
and demonstrating the evaluation pipeline.

**SciFact** — Real scientific claim-based dataset from the BEIR benchmark.
~5,183 documents from PubMed abstracts. Run `python scifact_import.py` to
download and prepare it.

---

## 📐 Metrics

### Precision@k
```
P@k = (Relevant docs in top-k) / k
```

### Recall@k
```
Recall@k = (Relevant retrieved in top-k) / (Total relevant in judged pool)
```
*Recall requires complete relevance assessments to be accurate.*

### AP / MAP
```
AP@k = Σ [P@i × rel(i)] / total_relevant
MAP  = mean AP over all queries
```

### NDCG@k
```
DCG@k  = Σ (2^rel_i − 1) / log₂(i+1)
NDCG@k = DCG@k / IDCG@k
```

---

## 🏷️ Human Judgment Grades (Search & Judge tab)

| Grade | Label            |
|-------|------------------|
| 3     | Highly Relevant  |
| 2     | Relevant         |
| 1     | Partially Relevant |
| 0     | Not Relevant     |

Binary threshold for Precision/Recall/MAP: **grade ≥ 2 = relevant**.

## 🏆 Benchmark Labels (Benchmark tab — SciFact only)

| Qrel | Meaning              |
|------|----------------------|
| 1    | Relevant (official)  |
| 0    | Not Relevant (official) |

Unjudged documents follow the **standard pooling assumption** and are treated as
non-relevant. Recall may be underestimated when qrels are incomplete.

---

## 📁 Project Files

```
human-relevance-judgment/
├── app.py                 — Streamlit interface
├── retrieval.py           — TF-IDF search engine
├── evaluation.py          — IR metrics
├── judgment_manager.py    — Dataset-scoped judgment persistence
├── benchmark.py           — SciFact qrel loading and benchmark eval
├── scifact_import.py      — One-time SciFact download script
├── test_suite.py          — Unit tests
├── documents.csv          — 40-document synthetic collection
├── documents_scifact.csv  — SciFact corpus (generated)
├── scifact_queries.csv    — SciFact test queries (generated)
├── scifact_qrels.csv      — Official qrels (generated)
├── judgments.csv          — Saved human judgments (auto-created)
├── requirements.txt
└── README.md
```
""")
