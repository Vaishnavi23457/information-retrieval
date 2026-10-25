"""
test_suite.py
-------------
Unit tests for the Human Relevance Judgment System.

Covers:
  1. Metric correctness (Precision, Recall, AP, MAP, NDCG)
  2. Query-specific qrels matching
  3. Missing / unjudged labels
  4. Dataset switching (judgment scoping)
  5. Ranking evaluation (benchmark_evaluate)
  6. judgment_manager round-trip (save → load → delete)

Run with:
    python test_suite.py
or:
    python -m pytest test_suite.py -v
"""

import os
import tempfile
import shutil
import unittest

# ── Patch JUDGMENTS_FILE so tests write to a temp file, not judgments.csv ──
import judgment_manager as jm

_ORIG_JFILE = jm.JUDGMENTS_FILE


def _patch_jfile(tmp_path: str):
    jm.JUDGMENTS_FILE = tmp_path


def _restore_jfile():
    jm.JUDGMENTS_FILE = _ORIG_JFILE


# ─────────────────────────────────────────────────────────────────────────────
# 1. Metric correctness
# ─────────────────────────────────────────────────────────────────────────────

from evaluation import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    mean_average_precision,
    evaluate_query,
    RELEVANCE_THRESHOLD,
)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Ranked list: D01(3), D02(0), D03(2), D04(1), D05(3)
        # Extra judged but not retrieved: D06(2), D07(3)
        self.docs = ["D01", "D02", "D03", "D04", "D05"]
        self.rel  = {"D01": 3, "D02": 0, "D03": 2, "D04": 1, "D05": 3,
                     "D06": 2, "D07": 3}
        # Binary relevant: D01, D03, D05, D06, D07 → 5 total
        # Retrieved relevant in top-5: D01, D03, D05 → 3

    # ── Precision ────────────────────────────────────────────────────────────

    def test_precision_at_5(self):
        p = precision_at_k(self.docs, self.rel, k=5)
        self.assertAlmostEqual(p, 3 / 5)

    def test_precision_at_3(self):
        # Top-3: D01(R), D02(NR), D03(R) → 2/3
        p = precision_at_k(self.docs, self.rel, k=3)
        self.assertAlmostEqual(p, 2 / 3)

    def test_precision_empty_list(self):
        self.assertEqual(precision_at_k([], self.rel, k=5), 0.0)

    def test_precision_no_relevant(self):
        p = precision_at_k(self.docs, {"D01": 0, "D02": 0}, k=5)
        self.assertEqual(p, 0.0)

    # ── Recall ───────────────────────────────────────────────────────────────

    def test_recall_at_5(self):
        r = recall_at_k(self.docs, self.rel, k=5)
        self.assertAlmostEqual(r, 3 / 5)  # 3 retrieved / 5 total relevant

    def test_recall_at_3(self):
        r = recall_at_k(self.docs, self.rel, k=3)
        self.assertAlmostEqual(r, 2 / 5)

    def test_recall_no_relevant_in_pool(self):
        r = recall_at_k(self.docs, {"D01": 0, "D02": 0}, k=5)
        self.assertEqual(r, 0.0)

    # ── AP ───────────────────────────────────────────────────────────────────

    def test_ap_perfect_ranking(self):
        # All retrieved docs are relevant, all in top positions
        docs = ["D01", "D03", "D05"]
        rel  = {"D01": 3, "D03": 2, "D05": 3}
        ap = average_precision(docs, rel, k=3)
        self.assertAlmostEqual(ap, 1.0)

    def test_ap_no_relevant(self):
        ap = average_precision(self.docs, {"D01": 0}, k=5)
        self.assertEqual(ap, 0.0)

    def test_ap_partial(self):
        ap = average_precision(self.docs, self.rel, k=5)
        # Relevant hits at ranks 1, 3, 5:
        # P@1=1/1, P@3=2/3, P@5=3/5; sum / 5 total_relevant
        expected = (1.0 + 2 / 3 + 3 / 5) / 5
        self.assertAlmostEqual(ap, expected, places=6)

    # ── MAP ──────────────────────────────────────────────────────────────────

    def test_map_single_query(self):
        ap = average_precision(self.docs, self.rel, k=5)
        m  = mean_average_precision(
            [{"ranked_doc_ids": self.docs, "relevance_dict": self.rel, "k": 5}]
        )
        self.assertAlmostEqual(m, ap)

    def test_map_two_queries(self):
        q2d = ["D06", "D01"]
        q2r = {"D06": 3, "D01": 2}
        m = mean_average_precision([
            {"ranked_doc_ids": self.docs, "relevance_dict": self.rel, "k": 5},
            {"ranked_doc_ids": q2d,       "relevance_dict": q2r,       "k": 2},
        ])
        ap1 = average_precision(self.docs, self.rel, k=5)
        ap2 = average_precision(q2d, q2r, k=2)
        self.assertAlmostEqual(m, (ap1 + ap2) / 2)

    def test_map_empty(self):
        self.assertEqual(mean_average_precision([]), 0.0)

    # ── NDCG ─────────────────────────────────────────────────────────────────

    def test_ndcg_perfect(self):
        # Ideal order for grades {3,3,2}: [3,3,2]
        docs = ["D01", "D05", "D03"]
        rel  = {"D01": 3, "D05": 3, "D03": 2}
        ndcg = ndcg_at_k(docs, rel, k=3)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_ndcg_worst_order(self):
        # Grades available: {3,2} but retrieved in reversed order
        docs = ["D03", "D01"]
        rel  = {"D01": 3, "D03": 2}
        ndcg = ndcg_at_k(docs, rel, k=2)
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)

    def test_ndcg_no_relevant(self):
        ndcg = ndcg_at_k(self.docs, {"D01": 0}, k=5)
        self.assertEqual(ndcg, 0.0)

    def test_ndcg_clamped_to_one(self):
        docs = ["D01"]
        rel  = {"D01": 3}
        ndcg = ndcg_at_k(docs, rel, k=1)
        self.assertLessEqual(ndcg, 1.0)

    # ── evaluate_query convenience wrapper ───────────────────────────────────

    def test_evaluate_query_keys(self):
        result = evaluate_query(self.docs, self.rel, k=5)
        self.assertSetEqual(set(result.keys()), {"precision", "recall", "ap", "ndcg"})

    def test_evaluate_query_range(self):
        result = evaluate_query(self.docs, self.rel, k=5)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Qrels matching
# ─────────────────────────────────────────────────────────────────────────────

from benchmark import benchmark_evaluate, multi_query_map


class TestQrelsMatching(unittest.TestCase):

    def setUp(self):
        # Simulated SciFact-style qrels: binary 0/1
        self.qrels = {
            "5678": 1,
            "1234": 1,
            "9999": 0,
        }
        self.ranked = ["5678", "9999", "1234", "0000", "1111"]
        # 0000, 1111 are not in qrels → unjudged

    def test_precision_with_binary_qrels(self):
        # SciFact qrels use 0/1.  benchmark_evaluate scales 1→2 so the
        # standard RELEVANCE_THRESHOLD=2 applies correctly.
        # Direct call to precision_at_k with raw 0/1 labels returns 0
        # (documented behaviour — use benchmark_evaluate for qrel data).
        p_raw = precision_at_k(self.ranked, self.qrels, k=3)
        self.assertEqual(p_raw, 0.0)

        # Via benchmark_evaluate (which scales labels): top-3 → 5678(R), 9999(NR), 1234(R) = 2/3
        result = benchmark_evaluate(self.ranked, self.qrels, k=3)
        self.assertAlmostEqual(result["precision"], 2 / 3)

    def test_recall_with_binary_qrels(self):
        # Direct call: raw qrel=1 < threshold → 0
        r_raw = recall_at_k(self.ranked, self.qrels, k=2)
        self.assertEqual(r_raw, 0.0)

        # Via benchmark_evaluate: top-2 → 5678(R), 9999(NR); 2 relevant total → 1/2
        result = benchmark_evaluate(self.ranked, self.qrels, k=2)
        self.assertAlmostEqual(result["recall"], 1 / 2)

    def test_benchmark_precision_with_elevated_qrels(self):
        # Confirm scaling logic: qrel*2 maps 1→2 meeting the threshold
        elevated = {k: (v * 2) for k, v in self.qrels.items()}
        p = precision_at_k(self.ranked, elevated, k=3)
        self.assertAlmostEqual(p, 2 / 3)

    def test_benchmark_recall_with_elevated_qrels(self):
        elevated = {k: (v * 2) for k, v in self.qrels.items()}
        r = recall_at_k(self.ranked, elevated, k=2)
        self.assertAlmostEqual(r, 1 / 2)

    def test_benchmark_evaluate_structure(self):
        result = benchmark_evaluate(self.ranked, self.qrels, k=5)
        for key in ("precision", "recall", "ap", "ndcg",
                    "relevant_in_qrels", "judged_in_top_k",
                    "unjudged_in_top_k", "recall_note"):
            self.assertIn(key, result)

    def test_benchmark_unjudged_count(self):
        result = benchmark_evaluate(self.ranked, self.qrels, k=5)
        # 0000, 1111 are unjudged in top-5
        self.assertEqual(result["unjudged_in_top_k"], 2)

    def test_benchmark_judged_count(self):
        result = benchmark_evaluate(self.ranked, self.qrels, k=5)
        self.assertEqual(result["judged_in_top_k"], 3)

    def test_recall_note_present(self):
        result = benchmark_evaluate(self.ranked, self.qrels, k=5)
        self.assertIsInstance(result["recall_note"], str)
        self.assertGreater(len(result["recall_note"]), 10)

    def test_multi_query_map(self):
        r1 = {"ranked_doc_ids": self.ranked,       "qrels_dict": self.qrels}
        r2 = {"ranked_doc_ids": ["5678", "1234"],  "qrels_dict": self.qrels}
        m = multi_query_map([r1, r2], k=5)
        self.assertGreaterEqual(m, 0.0)
        self.assertLessEqual(m, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Missing / unjudged labels
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingLabels(unittest.TestCase):

    def test_precision_all_unjudged(self):
        # None of the retrieved docs are in the relevance dict
        p = precision_at_k(["A", "B", "C"], {}, k=3)
        self.assertEqual(p, 0.0)

    def test_recall_all_unjudged(self):
        r = recall_at_k(["A", "B", "C"], {}, k=3)
        self.assertEqual(r, 0.0)

    def test_ndcg_all_unjudged(self):
        ndcg = ndcg_at_k(["A", "B", "C"], {}, k=3)
        self.assertEqual(ndcg, 0.0)

    def test_ap_mixed_judged(self):
        # Some docs judged, some not. Unjudged treated as grade=0 (non-relevant).
        docs = ["A", "B", "C", "D"]
        rel  = {"A": 3, "C": 2}      # B and D have no entry
        ap = average_precision(docs, rel, k=4)
        # total_relevant = 2; hits at rank 1 (A) and rank 3 (C)
        expected = (1.0 + 2 / 3) / 2
        self.assertAlmostEqual(ap, expected, places=6)

    def test_benchmark_with_no_qrels(self):
        result = benchmark_evaluate(["A", "B", "C"], {}, k=3)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["relevant_in_qrels"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset switching — judgment scoping
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetScoping(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".csv")
        _patch_jfile(self.tmp)

    def tearDown(self):
        _restore_jfile()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _make_judgments(self, grade: int):
        return {
            "D01": {"judgment": grade, "title": "Doc 1", "rank": 1},
            "D02": {"judgment": grade, "title": "Doc 2", "rank": 2},
        }

    def test_separate_storage_per_dataset(self):
        jm.save_judgments("cancer treatment", self._make_judgments(3), dataset="synthetic")
        jm.save_judgments("cancer treatment", self._make_judgments(1), dataset="scifact")

        syn_rel = jm.get_relevance_dict("cancer treatment", dataset="synthetic")
        sci_rel = jm.get_relevance_dict("cancer treatment", dataset="scifact")

        self.assertEqual(syn_rel["D01"], 3, "synthetic grade should be 3")
        self.assertEqual(sci_rel["D01"], 1, "scifact grade should be 1")

    def test_list_queries_filtered_by_dataset(self):
        jm.save_judgments("query A", self._make_judgments(2), dataset="synthetic")
        jm.save_judgments("query B", self._make_judgments(2), dataset="scifact")

        syn_list = jm.list_judged_queries(dataset="synthetic")
        sci_list = jm.list_judged_queries(dataset="scifact")

        self.assertEqual(len(syn_list), 1)
        self.assertEqual(syn_list.iloc[0]["query_text"], "query A")
        self.assertEqual(len(sci_list), 1)
        self.assertEqual(sci_list.iloc[0]["query_text"], "query B")

    def test_delete_removes_only_target_dataset(self):
        jm.save_judgments("shared query", self._make_judgments(2), dataset="synthetic")
        jm.save_judgments("shared query", self._make_judgments(2), dataset="scifact")
        jm.delete_query("shared query", dataset="synthetic")

        syn_rel = jm.get_relevance_dict("shared query", dataset="synthetic")
        sci_rel = jm.get_relevance_dict("shared query", dataset="scifact")

        self.assertEqual(syn_rel, {}, "synthetic judgments should be deleted")
        self.assertNotEqual(sci_rel, {}, "scifact judgments should remain")

    def test_overwrite_same_dataset(self):
        jm.save_judgments("overwrite test", self._make_judgments(3), dataset="synthetic")
        jm.save_judgments("overwrite test", self._make_judgments(0), dataset="synthetic")
        rel = jm.get_relevance_dict("overwrite test", dataset="synthetic")
        self.assertEqual(rel["D01"], 0, "overwrite should store grade 0")

    def test_get_ranked_doc_ids_correct_order(self):
        judgments = {
            "D02": {"judgment": 2, "title": "Doc 2", "rank": 1},
            "D01": {"judgment": 3, "title": "Doc 1", "rank": 2},
        }
        jm.save_judgments("order test", judgments, dataset="synthetic")
        ranked = jm.get_ranked_doc_ids("order test", dataset="synthetic")
        self.assertEqual(ranked, ["D02", "D01"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Ranking evaluation end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestRankingEvaluation(unittest.TestCase):

    def test_ideal_ranking_gives_ndcg_one(self):
        # Perfect ranking: most relevant first
        docs = ["A", "B", "C"]
        rel  = {"A": 3, "B": 2, "C": 1}
        ndcg = ndcg_at_k(docs, rel, k=3)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_reversed_ranking_lower_ndcg(self):
        docs_ideal    = ["A", "B", "C"]
        docs_reversed = ["C", "B", "A"]
        rel = {"A": 3, "B": 2, "C": 1}
        ndcg_ideal    = ndcg_at_k(docs_ideal,    rel, k=3)
        ndcg_reversed = ndcg_at_k(docs_reversed, rel, k=3)
        self.assertGreater(ndcg_ideal, ndcg_reversed)

    def test_precision_decreases_at_larger_k_with_noise(self):
        # First 3 are relevant, next 7 are not
        docs = [f"R{i}" for i in range(3)] + [f"N{i}" for i in range(7)]
        rel  = {f"R{i}": 2 for i in range(3)}
        p3  = precision_at_k(docs, rel, k=3)
        p10 = precision_at_k(docs, rel, k=10)
        self.assertGreaterEqual(p3, p10)

    def test_recall_increases_with_k(self):
        docs = ["A", "B", "C", "D", "E"]
        rel  = {"A": 2, "C": 3, "E": 2}
        r3 = recall_at_k(docs, rel, k=3)
        r5 = recall_at_k(docs, rel, k=5)
        self.assertLessEqual(r3, r5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Retrieval engine smoke test
# ─────────────────────────────────────────────────────────────────────────────

from retrieval import RetrievalEngine


class TestRetrievalEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = RetrievalEngine("documents.csv")

    def test_search_returns_dataframe(self):
        import pandas as pd
        results = self.engine.search("machine learning healthcare", top_k=5)
        self.assertIsInstance(results, pd.DataFrame)

    def test_search_correct_columns(self):
        results = self.engine.search("deep learning", top_k=3)
        expected = {"rank", "doc_id", "title", "content", "similarity_score", "snippet"}
        self.assertTrue(expected.issubset(set(results.columns)))

    def test_search_top_k_respected(self):
        results = self.engine.search("neural network", top_k=7)
        self.assertLessEqual(len(results), 7)

    def test_empty_query_returns_empty(self):
        results = self.engine.search("   ")
        self.assertEqual(len(results), 0)

    def test_ranks_are_sequential(self):
        results = self.engine.search("healthcare AI", top_k=5)
        ranks = results["rank"].tolist()
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_similarity_scores_descending(self):
        results = self.engine.search("cancer prediction", top_k=5)
        scores = results["similarity_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_get_all_documents(self):
        docs = self.engine.get_all_documents()
        self.assertGreater(len(docs), 0)
        self.assertIn("doc_id", docs.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    classes = [
        TestMetrics,
        TestQrelsMatching,
        TestMissingLabels,
        TestDatasetScoping,
        TestRankingEvaluation,
        TestRetrievalEngine,
    ]
    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    passed = total - len(result.failures) - len(result.errors)
    print(f"\n{'='*60}")
    print(f"  {passed}/{total} tests passed")
    if result.failures or result.errors:
        print("  FAILED — see details above.")
    else:
        print("  ALL TESTS PASSED.")
    print("="*60)
