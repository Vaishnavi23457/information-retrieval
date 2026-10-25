"""
retrieval.py
------------
TF-IDF based document retrieval engine.
Handles text preprocessing, vectorisation, and cosine-similarity ranking.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# ──────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval engine
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Builds a TF-IDF index over a document collection and ranks documents
    against free-text queries using cosine similarity.
    """

    def __init__(self, documents_path: str = "documents.csv"):
        self.documents_path = documents_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self._load_and_index()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_and_index(self) -> None:
        """Load CSV and build the TF-IDF index."""
        self.df = pd.read_csv(self.documents_path)
        self._validate_columns()

        # Combine title + content for richer representation
        self.df["combined"] = (
            self.df["title"].fillna("") + " " + self.df["content"].fillna("")
        ).apply(preprocess_text)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,    # log-normalise term frequencies
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined"])

    def _validate_columns(self) -> None:
        required = {"doc_id", "title", "content"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"documents.csv is missing columns: {missing}. "
                "Expected: doc_id, title, content"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Retrieve and rank the top-k most similar documents for *query*.

        Returns a DataFrame with columns:
            rank, doc_id, title, content, similarity_score, snippet
        """
        if not query.strip():
            return pd.DataFrame()

        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices (highest scores first)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "content": row["content"],
                    "similarity_score": round(float(scores[idx]), 4),
                    "snippet": _make_snippet(row["content"], max_chars=200),
                }
            )

        return pd.DataFrame(results)

    def get_all_documents(self) -> pd.DataFrame:
        """Return the full document collection."""
        return self.df[["doc_id", "title", "content"]].copy()

    def reload(self) -> None:
        """Re-index documents (useful if documents.csv changes at runtime)."""
        self._load_and_index()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _make_snippet(text: str, max_chars: int = 200) -> str:
    """Return the first *max_chars* characters followed by '…' if truncated."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
