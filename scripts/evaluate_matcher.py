#!/usr/bin/env python3
"""Evaluate an existing matcher on a held-out JSONL file without retraining."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import brier_score_loss, classification_report, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/criterion_matcher.joblib"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    model = joblib.load(args.model)
    texts = [f"{row['requirement']} [SEP] {row['evidence']}" for row in rows]
    labels = [int(row["label"]) for row in rows]
    probabilities = model.predict_proba(texts)[:, 1]
    predictions = (probabilities >= 0.70).astype(int)
    report = classification_report(labels, predictions, zero_division=0, output_dict=True)
    result = {"rows": len(rows), "threshold": 0.70, "classification_report": report,
              "brier_score": round(brier_score_loss(labels, probabilities), 4)}
    if len(set(labels)) == 2:
        result["roc_auc"] = round(roc_auc_score(labels, probabilities), 4)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
