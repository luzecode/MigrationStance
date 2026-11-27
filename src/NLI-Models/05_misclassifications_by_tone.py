"""Summarize model misclassifications by actual stance, predicted stance, and tone."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nli"


def main():
    parser = argparse.ArgumentParser(
        description="Count model misclassifications by tone and stance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prediction CSV containing nli_label, predicted_nli_label, stance_3, and tone.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Prefix for the summary and misclassified-row CSV files.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use for the binary Political DEBATE model (entailment vs. not_entailment).",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"nli_label", "predicted_nli_label", "stance_3", "tone"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns in {args.input}: {sorted(missing)}")

    actual = data["nli_label"].astype(str).str.strip().str.lower()
    predicted = data["predicted_nli_label"].astype(str).str.strip().str.lower()
    if args.binary:
        actual = actual.where(actual.eq("entailment"), "not_entailment")
        predicted_stance = predicted.map({
            "entailment": "pro",
            "not_entailment": "contra_or_neutral",
        })
    else:
        predicted_stance = predicted.map({
            "entailment": "pro",
            "neutral": "neutral",
            "contradiction": "contra",
        })
    if predicted_stance.isna().any():
        unknown = sorted(predicted[predicted_stance.isna()].unique())
        raise ValueError(f"Unrecognized predicted labels: {unknown}")

    data["misclassified"] = predicted.ne(actual)
    data["actual_stance"] = data["stance_3"]
    data["predicted_stance"] = predicted_stance
    groups = data.groupby(["stance_3", "tone"], dropna=False)["misclassified"]
    summary = groups.agg(
        total_samples="size",
        misclassified_count="sum",
        misclassification_rate="mean",
    ).reset_index()
    by_tone = data.groupby("tone", dropna=False)["misclassified"].agg(
        total_samples="size",
        misclassified_count="sum",
        misclassification_rate="mean",
    ).reset_index()
    by_tone.insert(0, "stance_3", "all")
    summary = pd.concat([summary, by_tone], ignore_index=True)

    stance_tone_breakdown = data.groupby(
        ["actual_stance", "predicted_stance", "tone"], dropna=False
    )["misclassified"].agg(
        total_samples="size",
        misclassified_count="sum",
        misclassification_rate="mean",
    ).reset_index()

    prefix = args.output_prefix or args.input.with_name(
        args.input.stem + "_tone_errors"
    )
    summary_path = prefix.with_name(prefix.name + "_summary.csv")
    mistakes_path = prefix.with_name(prefix.name + "_misclassified.csv")
    breakdown_path = prefix.with_name(prefix.name + "_stance_tone_breakdown.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    data.loc[data["misclassified"]].to_csv(mistakes_path, index=False)
    stance_tone_breakdown.to_csv(breakdown_path, index=False)

    print(summary.to_string(index=False))
    print("\nActual stance, predicted stance, and tone:")
    print(stance_tone_breakdown.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    print(f"Misclassified rows saved to: {mistakes_path}")
    print(f"Stance/tone breakdown saved to: {breakdown_path}")


if __name__ == "__main__":
    main()
