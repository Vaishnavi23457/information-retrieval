"""
evaluation.py
-------------
IR evaluation metrics: Precision@k, Recall@k, MAP, NDCG@k.

Binary relevance threshold
--------------------------
Grades 2 (Relevant) and 3 (Highly Relevant) count as *relevant* for
Precision, Recall, and MAP.
Grade 1 (Partially Relevant) and 0 (Not Relevant) count as *non-relevant*.
NDCG uses all four grades as graded relevance (0-3).
"""

import math
from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 2   # grades >= this value are "relevant" for binary metrics


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _is_relevant(grade: int) -> bool:
    return int(grade) >= RELEVANCE_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# Precision @ k
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(ranked_doc_ids: List[str],
                   relevance_dict: Dict[str, int],
                   k: Optional[int] = None) -> float:
    """
    P@k = (number of relevant docs in top-k) / k

    Parameters
    ----------
    ranked_doc_ids : list of doc_ids in retrieval order (rank 1 first)
    relevance_dict : {doc_id: grade}  for all judged docs
    k              : cut-off rank (defaults to len(ranked_doc_ids))
    """
    if not ranked_doc_ids:
        return 0.0

    k = k or len(ranked_doc_ids)
    top_k = ranked_doc_ids[:k]
    relevant_count = sum(
        1 for doc_id in top_k
        if _is_relevant(relevance_dict.get(doc_id, 0))
    )
    return relevant_count / k


# ──────────────────────────────────────────────────────────────────────────────
# Recall @ k
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(ranked_doc_ids: List[str],
                relevance_dict: Dict[str, int],
                k: Optional[int] = None) -> float:
    """
    Recall@k = (relevant docs in top-k) / (total relevant docs in judged pool)

    If there are no relevant documents in the judged pool, returns 0.0.
    """
    if not ranked_doc_ids:
        return 0.0

    total_relevant = sum(1 for g in relevance_dict.values() if _is_relevant(g))
    if total_relevant == 0:
        return 0.0

    k = k or len(ranked_doc_ids)
    top_k = ranked_doc_ids[:k]
    retrieved_relevant = sum(
        1 for doc_id in top_k
        if _is_relevant(relevance_dict.get(doc_id, 0))
    )
    return retrieved_relevant / total_relevant


# ──────────────────────────────────────────────────────────────────────────────
# Average Precision (AP) — used internally for MAP
# ──────────────────────────────────────────────────────────────────────────────

def average_precision(ranked_doc_ids: List[str],
                      relevance_dict: Dict[str, int],
                      k: Optional[int] = None) -> float:
    """
    AP@k = Σ [P@i × rel(i)] / total_relevant

    rel(i) = 1 if doc at rank i is relevant (grade ≥ threshold), else 0.
    """
    if not ranked_doc_ids:
        return 0.0

    total_relevant = sum(1 for g in relevance_dict.values() if _is_relevant(g))
    if total_relevant == 0:
        return 0.0

    k = k or len(ranked_doc_ids)
    top_k = ranked_doc_ids[:k]

    running_relevant = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(top_k, start=1):
        if _is_relevant(relevance_dict.get(doc_id, 0)):
            running_relevant += 1
            sum_precision += running_relevant / i   # P@i

    return sum_precision / total_relevant


# ──────────────────────────────────────────────────────────────────────────────
# Mean Average Precision (MAP)
# ──────────────────────────────────────────────────────────────────────────────

def mean_average_precision(queries: List[Dict]) -> float:
    """
    MAP = mean of AP over all queries.

    Parameters
    ----------
    queries : list of dicts, each with:
        "ranked_doc_ids" : list[str]
        "relevance_dict" : dict[str, int]
        "k"              : int (optional, defaults to full list)
    """
    if not queries:
        return 0.0

    ap_values = []
    for q in queries:
        ap = average_precision(
            q["ranked_doc_ids"],
            q["relevance_dict"],
            q.get("k"),
        )
        ap_values.append(ap)

    return sum(ap_values) / len(ap_values)


# ──────────────────────────────────────────────────────────────────────────────
# DCG / NDCG @ k
# ──────────────────────────────────────────────────────────────────────────────

def _dcg_at_k(grades: List[int], k: int) -> float:
    """
    DCG@k = Σ (2^rel_i - 1) / log2(i + 1)   for i in 1..k
    """
    dcg = 0.0
    for i, grade in enumerate(grades[:k], start=1):
        dcg += (2 ** grade - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_doc_ids: List[str],
              relevance_dict: Dict[str, int],
              k: Optional[int] = None) -> float:
    """
    NDCG@k = DCG@k / IDCG@k

    Uses graded relevance (0-3) from *relevance_dict*.
    Returns 0.0 if IDCG = 0 (no relevant documents judged).
    """
    if not ranked_doc_ids:
        return 0.0

    k = k or len(ranked_doc_ids)
    top_k = ranked_doc_ids[:k]

    # Grades in retrieval order
    actual_grades = [int(relevance_dict.get(doc_id, 0)) for doc_id in top_k]

    # Ideal: sort all judged grades descending, take top-k
    all_grades = sorted(relevance_dict.values(), reverse=True)
    ideal_grades = [int(g) for g in all_grades[:k]]

    idcg = _dcg_at_k(ideal_grades, k)
    if idcg == 0:
        return 0.0

    dcg = _dcg_at_k(actual_grades, k)
    return min(dcg / idcg, 1.0)   # clamp to [0, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: evaluate one query at multiple cut-offs
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_query(ranked_doc_ids: List[str],
                   relevance_dict: Dict[str, int],
                   k: int = 10) -> Dict[str, float]:
    """
    Compute all four metrics for a single query at cut-off k.

    Returns
    -------
    dict with keys: precision, recall, ap, ndcg
    """
    return {
        "precision": precision_at_k(ranked_doc_ids, relevance_dict, k),
        "recall": recall_at_k(ranked_doc_ids, relevance_dict, k),
        "ap": average_precision(ranked_doc_ids, relevance_dict, k),
        "ndcg": ndcg_at_k(ranked_doc_ids, relevance_dict, k),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test (run with: python evaluation.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Worked example from the project spec
    # Relevance grades for 5 retrieved docs: [3, 0, 2, 1, 3]
    # Two additional relevant docs not retrieved (total relevant = 5)

    docs = ["D01", "D02", "D03", "D04", "D05"]
    rel  = {"D01": 3, "D02": 0, "D03": 2, "D04": 1, "D05": 3,
            "D06": 2, "D07": 3}   # D06, D07 judged but not retrieved

    k = 5

    p   = precision_at_k(docs, rel, k)
    r   = recall_at_k(docs, rel, k)
    ap  = average_precision(docs, rel, k)
    ndcg= ndcg_at_k(docs, rel, k)

    print(f"Precision@{k} : {p:.4f}")
    print(f"Recall@{k}    : {r:.4f}")
    print(f"AP@{k}        : {ap:.4f}")
    print(f"NDCG@{k}      : {ndcg:.4f}")

    # MAP over two queries
    q2_docs = ["D06", "D01", "D07"]
    q2_rel  = {"D01": 2, "D06": 3, "D07": 2, "D08": 1}
    map_val = mean_average_precision([
        {"ranked_doc_ids": docs, "relevance_dict": rel, "k": k},
        {"ranked_doc_ids": q2_docs, "relevance_dict": q2_rel, "k": 3},
    ])
    print(f"MAP           : {map_val:.4f}")
