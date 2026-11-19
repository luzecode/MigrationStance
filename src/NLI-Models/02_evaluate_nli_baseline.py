"""Evaluate an untuned generic NLI model on the stance test set.

to run the baseline on the whoel dataset:

.\.venv-labelstudio\Scripts\python.exe .\src\NLI-Models\02_evaluate_nli_baseline.py `
  --input ".\data\nli\nli_dataset.csv" `
  --output ".\data\nli\nli_baseline_full_results.json" `
  --predictions ".\data\nli\nli_baseline_full_predictions.csv"
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nli"
MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
LABELS = ("entailment", "neutral", "contradiction")
SEED = 20261105


def normalise_label(label):
    value = str(label).strip().lower()
    aliases = {
        "entailment": "entailment",
        "contradiction": "contradiction",
        "neutral": "neutral",
        "label_0": "contradiction",
        "label_1": "neutral",
        "label_2": "entailment",
    }
    return aliases.get(value, value)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the untuned generic mDeBERTa NLI model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA / "nli_test.csv",
        help="CSV containing premise, hypothesis, and nli_label columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA / "nli_baseline_results.json",
        help="Path for JSON evaluation results.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DATA / "nli_baseline_predictions.csv",
        help="Path for row-level predictions.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    test = pd.read_csv(args.input)
    required = {"premise", "hypothesis", "nli_label"}
    missing = required.difference(test.columns)
    if missing:
        raise KeyError(f"Missing required columns in {args.input}: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    predictions = []
    for start in range(0, len(test), 16):
        batch = test.iloc[start:start + 16]
        encoded = tokenizer(
            batch["premise"].tolist(),
            batch["hypothesis"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    id2label = model.config.id2label
    predicted_labels = [
        normalise_label(id2label.get(index, id2label.get(str(index), "")))
        for index in predictions
    ]
    actual_labels = test["nli_label"].map(normalise_label).tolist()

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=list(LABELS),
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(actual_labels, predicted_labels, labels=list(LABELS))

    test_output = test.copy()
    test_output["predicted_nli_label"] = predicted_labels
    test_output.to_csv(args.predictions, index=False)

    results = {
        "model": MODEL_NAME,
        "test_samples": len(test),
        "labels": list(LABELS),
        "classification_report": report,
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": matrix.tolist(),
        },
    }
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
