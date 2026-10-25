"""
scifact_import.py
-----------------
Downloads the BEIR SciFact dataset and converts it into the three CSV files
the application expects:

    documents_scifact.csv   — doc_id, title, content, url
    scifact_queries.csv     — query_id, query_text
    scifact_qrels.csv       — query_id, doc_id, relevance

Run once before launching the app:
    python scifact_import.py

Requirements: pip install beir
(beir is already listed in requirements.txt)
"""

import os
import json
import csv
import sys


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  Written {len(rows):,} rows → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main import
# ──────────────────────────────────────────────────────────────────────────────

def import_scifact(out_dir: str = ".") -> None:
    """
    Download SciFact via BEIR and write the three CSV files.
    Safe to re-run — skips download if the raw data folder already exists.
    """
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        print(
            "\n[ERROR] The 'beir' package is not installed.\n"
            "Run:  pip install beir\n"
            "Then re-run this script."
        )
        sys.exit(1)

    raw_dir = os.path.join(out_dir, "scifact_raw")

    # ── 1. Download (skipped if already present) ──────────────────────────────
    if not os.path.isdir(raw_dir):
        print("Downloading SciFact dataset from BEIR …")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
        zip_path = beir_util.download_and_unzip(url, out_dir)
        # BEIR unzips to <out_dir>/scifact — rename to raw_dir for clarity
        extracted = os.path.join(out_dir, "scifact")
        if os.path.isdir(extracted) and not os.path.isdir(raw_dir):
            os.rename(extracted, raw_dir)
        print("Download complete.")
    else:
        print(f"Raw data already present at '{raw_dir}' — skipping download.")

    # ── 2. Load with BEIR ─────────────────────────────────────────────────────
    print("Loading corpus, queries, and qrels …")
    corpus, queries, qrels = GenericDataLoader(data_folder=raw_dir).load(split="test")
    print(f"  Corpus : {len(corpus):,} documents")
    print(f"  Queries: {len(queries):,}")
    print(f"  Qrels  : {sum(len(v) for v in qrels.values()):,} judgments")

    # ── 3. Write documents_scifact.csv ────────────────────────────────────────
    doc_rows = []
    for doc_id, doc in corpus.items():
        doc_rows.append(
            {
                "doc_id": str(doc_id),
                "title": doc.get("title", "").strip(),
                "content": doc.get("text", "").strip(),
                "url": doc.get("url", ""),
            }
        )
    _write_csv(
        os.path.join(out_dir, "documents_scifact.csv"),
        doc_rows,
        ["doc_id", "title", "content", "url"],
    )

    # ── 4. Write scifact_queries.csv ──────────────────────────────────────────
    query_rows = [
        {"query_id": str(qid), "query_text": text.strip()}
        for qid, text in queries.items()
    ]
    _write_csv(
        os.path.join(out_dir, "scifact_queries.csv"),
        query_rows,
        ["query_id", "query_text"],
    )

    # ── 5. Write scifact_qrels.csv ────────────────────────────────────────────
    qrel_rows = []
    for qid, doc_dict in qrels.items():
        for doc_id, relevance in doc_dict.items():
            qrel_rows.append(
                {
                    "query_id": str(qid),
                    "doc_id": str(doc_id),
                    "relevance": int(relevance),   # 0 or 1 in SciFact
                }
            )
    _write_csv(
        os.path.join(out_dir, "scifact_qrels.csv"),
        qrel_rows,
        ["query_id", "doc_id", "relevance"],
    )

    print("\n✅  SciFact import complete. You can now launch the app.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import_scifact(out_dir=script_dir)
