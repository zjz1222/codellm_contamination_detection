"""Verify group isolation and equal-weight macro metrics without LLM inference."""
from pathlib import Path
import sys
import unittest

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment import grouped_scores


class GroupedEvaluationTests(unittest.TestCase):
    def test_groups_and_macro_are_not_pooled(self):
        rows = []
        for group, repeats in [("original", 1), ("Type I", 2), ("Type II", 3), ("Type_III", 4)]:
            for _ in range(repeats):
                for label in [0, 1]:
                    # Last group has reversed scores; unequal group sizes expose pooling.
                    score = 1-label if group == "Type_III" else label
                    rows.append({"type": group, "label": label, "code": str(score)})
        calls = []

        def predict(model, tokenizer, codes, max_length):
            calls.append(codes)
            return [float(code) for code in codes]

        def calculate(scores, labels):
            return pd.DataFrame([{"method": "synprune", "auc": f"{100*roc_auc_score(labels, scores)}%",
                                  "accuracy": "50%", "tpr01": "0%"}])

        table = grouped_scores({"eval": predict, "calculate": calculate}, None, None, rows, 4096, "synprune")
        self.assertEqual([len(x) for x in calls], [2, 4, 6, 8])
        self.assertEqual(table["Clone Type"].tolist(), ["original", "Type I", "Type II", "Type III", "Macro Average"])
        self.assertEqual(table.iloc[-1]["auc"], 0.75)
        self.assertNotEqual(table.iloc[-1]["auc"], roc_auc_score([r["label"] for r in rows], [float(r["code"]) for r in rows]))

    def test_missing_class_fails_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "Both labels"):
            grouped_scores({}, None, None, [{"type": "original", "code": "x", "label": 1}], 4096, "synprune")
