#!/usr/bin/env python3
"""Train and evaluate the RoleFit supervised evidence matcher.

Usage: python scripts/train_matcher.py data/labeled_pairs.jsonl
The output model is written to models/criterion_matcher.joblib only after a
grouped holdout evaluation succeeds.
"""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline


def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"job_id", "requirement", "evidence", "label"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Every JSONL row needs job_id, requirement, evidence, and label.")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError("The labeled data must contain both positive (1) and negative (0) examples.")
    return rows


def make_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=50000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    rows = load_rows(args.dataset)
    if len({row["job_id"] for row in rows}) < 4:
        raise ValueError("Use at least four distinct job_id groups for a meaningful holdout.")
    texts = [f"{row['requirement']} [SEP] {row['evidence']}" for row in rows]
    labels = [int(row["label"]) for row in rows]
    groups = [row["job_id"] for row in rows]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42).split(texts, labels, groups))
    model = make_pipeline()
    model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
    probabilities = model.predict_proba([texts[i] for i in test_idx])[:, 1]
    predictions = (probabilities >= 0.70).astype(int)
    y_test = [labels[i] for i in test_idx]
    metrics = {
        "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_jobs": len({groups[i] for i in train_idx}), "test_jobs": len({groups[i] for i in test_idx}),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision_at_0_70": round(precision_score(y_test, predictions, zero_division=0), 4),
        "recall_at_0_70": round(recall_score(y_test, predictions, zero_division=0), 4),
        "f1_at_0_70": round(f1_score(y_test, predictions, zero_division=0), 4),
        "brier_score": round(brier_score_loss(y_test, probabilities), 4),
        "classification_report": classification_report(y_test, predictions, zero_division=0, output_dict=True),
    }
    if len(set(y_test)) == 2:
        metrics["roc_auc"] = round(roc_auc_score(y_test, probabilities), 4)
    print(json.dumps(metrics, indent=2))
    out = Path("models/criterion_matcher.joblib")
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out)
    Path("models/criterion_matcher.metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {out} and its holdout metrics.")


if __name__ == "__main__":
    main()
