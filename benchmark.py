"""
benchmark.py
------------
Benchmark evaluation against official SciFact qrels.

Key design decisions
--------------------
- Qrel relevance labels (0/1) are kept SEPARATE from human judgments (0-3).
  They are never mixed.
- SciFact qrels use binary relevance: 1 = relevant, 0 = not relevant.
- Unjudged documents (not in qrels for a query) are treated as non-relevant
  for Precision and MAP, following the standard pooling assumption.  This
  assumption is documented explicitly wherever recall is displayed.
- NDCG uses qrel grades directly (0 or 1) without up-scaling.

Public API
----------
    load_queries()          → DataFrame(query_id, query_text)
    load_qrels()            → DataFrame(query_id, doc_id, relevance)
    get_qrels_for_query(qid)→ {doc_id: relevance}   (binary 0/1)
    benchmark_evaluate(ranked_doc_ids, qrels_dict, k) → dict of metrics
    multi_query_map(results_per_query, k) → float
"""

import os
import pandas as pd
from typing import Dict, List, Optional

from evaluation import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    mean_average_precision,
)


# ──────────────────────────────────────────────────────────────────────────────
# File paths (resolved relative to this script)
# ──────────────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES_FILE = os.path.join(_HERE, "scifact_queries.csv")
QRELS_FILE   = os.path.join(_HERE, "scifact_qrels.csv")
DOCS_FILE    = os.path.join(_HERE, "documents_scifact.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Availability check
# ──────────────────────────────────────────────────────────────────────────────

def scifact_available() -> bool:
    """Return True only when all three SciFact CSVs are present."""
    return all(os.path.isfile(p) for p in [QUERIES_FILE, QRELS_FILE, DOCS_FILE])


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_queries() -> pd.DataFrame:
    """
    Return DataFrame with columns: query_id (str), query_text (str).
    Raises FileNotFoundError if scifact_queries.csv is missing.
    """
    df = pd.read_csv(QUERIES_FILE, dtype={"query_id": str})
    df["query_id"]   = df["query_id"].str.strip()
    df["query_text"] = df["query_text"].str.strip()
    return df.reset_index(drop=True)


def load_qrels() -> pd.DataFrame:
    """
    Return DataFrame with columns: query_id (str), doc_id (str), relevance (int).
    """
    df = pd.read_csv(QRELS_FILE, dtype={"query_id": str, "doc_id": str})
    df["query_id"]  = df["query_id"].str.strip()
    df["doc_id"]    = df["doc_id"].str.strip()
    df["relevance"] = df["relevance"].astype(int)
    return df.reset_index(drop=True)


def get_qrels_for_query(query_id: str) -> Dict[str, int]:
    """
    Return {doc_id: relevance} for one query_id.
    Relevance values are binary (0 or 1) from the official SciFact qrels.
    Documents not listed in qrels are NOT included — callers must decide
    whether to treat them as 0 (standard pooling assumption).
    """
    df = load_qrels()
    subset = df[df["query_id"] == str(query_id)]
    return dict(zip(subset["doc_id"].astype(str), subset["relevance"].astype(int)))


def count_relevant_in_qrels(query_id: str) -> int:
    """Return the number of documents judged relevant (grade=1) for a query."""
    qrels = get_qrels_for_query(query_id)
    return sum(1 for v in qrels.values() if v >= 1)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark evaluation (single query)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_evaluate(
    ranked_doc_ids: List[str],
    qrels_dict: Dict[str, int],
    k: int = 10,
) -> Dict:
    """
    Evaluate a ranked list against official qrels.

    SciFact qrels use binary labels: 1 = relevant, 0 = not relevant.
    The shared metric functions use RELEVANCE_THRESHOLD = 2, so we scale
    binary labels: qrel 1 → grade 2 (Relevant), qrel 0 → grade 0.
    This keeps the metric functions unchanged while making benchmark
    evaluation work correctly with binary qrel data.

    Parameters
    ----------
    ranked_doc_ids : list of doc_ids in retrieval order (rank 1 first)
    qrels_dict     : {doc_id: relevance}  from official qrels (binary 0/1)
    k              : cut-off rank

    Returns
    -------
    dict with keys:
        precision, recall, ap, ndcg     — float metrics
        relevant_in_qrels               — int, total judged-relevant docs
        judged_in_top_k                 — int, top-k docs that have a qrel entry
        unjudged_in_top_k               — int, top-k docs WITHOUT a qrel entry
        recall_note                     — str, caveat about recall completeness
    """
    top_k = ranked_doc_ids[:k]
    judged_in_top_k   = sum(1 for d in top_k if d in qrels_dict)
    unjudged_in_top_k = k - judged_in_top_k
    relevant_in_qrels = sum(1 for v in qrels_dict.values() if v >= 1)

    # Scale binary qrel labels (0/1) → (0/2) so they meet RELEVANCE_THRESHOLD=2.
    # Grade 0 stays 0 (non-relevant); grade 1 becomes 2 (Relevant).
    # This does NOT fabricate human judgment grades — it is purely an
    # adapter between the official binary annotation convention and our
    # shared metric functions.
    scaled_qrels = {doc_id: (grade * 2) for doc_id, grade in qrels_dict.items()}

    # Standard pooling: unjudged docs treated as non-relevant (grade=0).
    # The full scaled_qrels is passed so total_relevant is correct.
    metrics = {
        "precision": precision_at_k(ranked_doc_ids, scaled_qrels, k),
        "recall":    recall_at_k(ranked_doc_ids, scaled_qrels, k),
        "ap":        average_precision(ranked_doc_ids, scaled_qrels, k),
        "ndcg":      ndcg_at_k(ranked_doc_ids, scaled_qrels, k),
        "relevant_in_qrels":   relevant_in_qrels,
        "judged_in_top_k":     judged_in_top_k,
        "unjudged_in_top_k":   unjudged_in_top_k,
        "recall_note": (
            f"Recall is computed over {relevant_in_qrels} judged-relevant document(s) "
            f"in the official qrels. {unjudged_in_top_k} of the top-{k} retrieved "
            "document(s) have no qrel entry and are treated as non-relevant "
            "(standard pooling assumption). If the qrels are incomplete, "
            "recall may be underestimated."
        ),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Multi-query MAP for benchmark
# ──────────────────────────────────────────────────────────────────────────────

def multi_query_map(results_per_query: List[Dict], k: int = 10) -> float:
    """
    Compute MAP across multiple benchmark queries.

    Parameters
    ----------
    results_per_query : list of dicts, each with:
        "ranked_doc_ids" : list[str]
        "qrels_dict"     : dict[str, int]

    Returns MAP as a float.
    """
    queries_for_map = [
        {
            "ranked_doc_ids": r["ranked_doc_ids"],
            "relevance_dict": r["qrels_dict"],
            "k": k,
        }
        for r in results_per_query
    ]
    return mean_average_precision(queries_for_map)
