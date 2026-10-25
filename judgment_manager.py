"""
judgment_manager.py
-------------------
Saves, loads, and queries human relevance judgments.

Judgments are persisted in judgments.csv with the schema:
    dataset | query_id | query_text | doc_id | title | rank | judgment | timestamp

The 'dataset' column scopes every judgment to its collection
("synthetic" or "scifact") so ratings cannot be mixed across datasets.

Backward compatibility
----------------------
All public functions accept a keyword argument  dataset="synthetic"
which is used as the default.  Existing callers that pass no dataset
argument continue to work unchanged.
"""

import os
import pandas as pd
from datetime import datetime


JUDGMENTS_FILE = "judgments.csv"

COLUMNS = [
    "dataset",
    "query_id",
    "query_text",
    "doc_id",
    "title",
    "rank",
    "judgment",      # 0 = Not Relevant, 1 = Partially, 2 = Relevant, 3 = Highly Relevant
    "timestamp",
]

LABEL_MAP = {
    3: "Highly Relevant",
    2: "Relevant",
    1: "Partially Relevant",
    0: "Not Relevant",
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv() -> pd.DataFrame:
    """Load existing judgments or return an empty DataFrame."""
    if os.path.exists(JUDGMENTS_FILE):
        try:
            df = pd.read_csv(JUDGMENTS_FILE, dtype={"doc_id": str, "query_id": str})
            # Back-fill 'dataset' column for rows saved before this column existed
            if "dataset" not in df.columns:
                df["dataset"] = "synthetic"
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[COLUMNS]
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)


def _save_csv(df: pd.DataFrame) -> None:
    df.to_csv(JUDGMENTS_FILE, index=False)


def _make_query_id(query_text: str, dataset: str = "synthetic") -> str:
    """Deterministic query ID scoped by dataset + normalised query text."""
    normalised = f"{dataset}::{query_text.strip().lower()}"
    return "Q" + str(abs(hash(normalised)) % 10_000_000).zfill(7)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def save_judgments(
    query_text: str,
    judgment_dict: dict,
    dataset: str = "synthetic",
) -> str:
    """
    Persist a batch of judgments for one query in one dataset.

    Parameters
    ----------
    query_text    : str  — the user's search query
    judgment_dict : dict — doc_id → {"judgment": int, "title": str, "rank": int}
    dataset       : str  — "synthetic" or "scifact" (default "synthetic")

    Returns
    -------
    str — the query_id assigned to this query
    """
    query_id  = _make_query_id(query_text, dataset)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    existing = _load_csv()

    # Drop previous judgments for the same dataset+query so re-ratings overwrite
    mask = (existing["query_id"] == query_id) & (existing["dataset"] == dataset)
    existing = existing[~mask]

    new_rows = []
    for doc_id, info in judgment_dict.items():
        new_rows.append(
            {
                "dataset":    dataset,
                "query_id":   query_id,
                "query_text": query_text,
                "doc_id":     str(doc_id),
                "title":      info.get("title", ""),
                "rank":       info.get("rank", 0),
                "judgment":   int(info.get("judgment", 0)),
                "timestamp":  timestamp,
            }
        )

    updated = pd.concat(
        [existing, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True
    )
    _save_csv(updated)
    return query_id


def load_judgments_for_query(
    query_text: str,
    dataset: str = "synthetic",
) -> pd.DataFrame:
    """Return the saved judgments DataFrame for one query+dataset (may be empty)."""
    query_id = _make_query_id(query_text, dataset)
    df = _load_csv()
    mask = (df["query_id"] == query_id) & (df["dataset"] == dataset)
    return df[mask].reset_index(drop=True)


def load_all_judgments() -> pd.DataFrame:
    """Return the complete judgments table (all datasets)."""
    return _load_csv()


def list_judged_queries(dataset: str = "synthetic") -> pd.DataFrame:
    """Return a summary of every query that has saved judgments for *dataset*."""
    df = _load_csv()
    if df.empty:
        return pd.DataFrame(
            columns=["query_id", "query_text", "num_judgments", "timestamp"]
        )
    subset = df[df["dataset"] == dataset]
    if subset.empty:
        return pd.DataFrame(
            columns=["query_id", "query_text", "num_judgments", "timestamp"]
        )
    summary = (
        subset.groupby(["query_id", "query_text"])
        .agg(num_judgments=("doc_id", "count"), timestamp=("timestamp", "max"))
        .reset_index()
    )
    return summary.sort_values("timestamp", ascending=False)


def get_relevance_dict(
    query_text: str,
    dataset: str = "synthetic",
) -> dict:
    """
    Return {doc_id: judgment_grade} for the given query+dataset.
    """
    df = load_judgments_for_query(query_text, dataset)
    if df.empty:
        return {}
    return dict(zip(df["doc_id"].astype(str), df["judgment"].astype(int)))


def get_ranked_doc_ids(
    query_text: str,
    dataset: str = "synthetic",
) -> list:
    """Return doc_ids in retrieval order for the given query+dataset."""
    df = load_judgments_for_query(query_text, dataset)
    if df.empty:
        return []
    return df.sort_values("rank")["doc_id"].astype(str).tolist()


def delete_query(
    query_text: str,
    dataset: str = "synthetic",
) -> bool:
    """Remove all judgments for a query+dataset. Returns True if anything was deleted."""
    query_id = _make_query_id(query_text, dataset)
    df = _load_csv()
    before = len(df)
    mask = (df["query_id"] == query_id) & (df["dataset"] == dataset)
    df = df[~mask]
    if len(df) < before:
        _save_csv(df)
        return True
    return False


def label_for(grade: int) -> str:
    """Human-readable label for an integer grade."""
    return LABEL_MAP.get(int(grade), "Unknown")
