"""Compare NLI error rates when tone aligns with stance and when it does not.

Aligned: pro + negativ, contra + positiv (typical discourse).
Misaligned: pro + positiv, contra + negativ.
Neutral stance and neutral/gemischt tone are left out.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nli"
OUTPUT = ROOT / "data" / "results" / "tone_bias.csv"
# prediction file -> whether the model is binary (entailment / not_entailment)
MODELS = {
    "nli_baseline_predictions.csv": False,
    "nli_political_debate_predictions.csv": True,
}
ALIGNMENT = {
    ("pro", "negativ"): "aligned",
    ("contra", "positiv"): "aligned",
    ("pro", "positiv"): "misaligned",
    ("contra", "negativ"): "misaligned",
}


def main():
    rows = []
    for file, binary in MODELS.items():
        if not (DATA / file).exists():
            print(f"Skipping {file}: run the evaluation script first.")
            continue
        data = pd.read_csv(DATA / file)
        gold = data["nli_label"]
        if binary:
            gold = gold.where(gold.eq("entailment"), "not_entailment")
        data["error"] = data["predicted_nli_label"] != gold
        data["alignment"] = [
            ALIGNMENT.get((stance, tone)) for stance, tone in zip(data["stance_3"], data["tone"])
        ]

        model_rows = []
        for stance in ["pro", "contra"]:
            subset = data[data["stance_3"] == stance]
            aligned = subset.loc[subset["alignment"] == "aligned", "error"]
            misaligned = subset.loc[subset["alignment"] == "misaligned", "error"]
            table = [
                [aligned.sum(), len(aligned) - aligned.sum()],
                [misaligned.sum(), len(misaligned) - misaligned.sum()],
            ]
            model_rows.append({
                "model": file.removesuffix("_predictions.csv"),
                "stance": stance,
                "n_aligned": len(aligned),
                "error_aligned": aligned.mean(),
                "n_misaligned": len(misaligned),
                "error_misaligned": misaligned.mean(),
                "difference": misaligned.mean() - aligned.mean(),
                "p_value": fisher_exact(table).pvalue,
            })
        # Average over stances, so the stance mix of the groups does not drive the result.
        model_rows.append({
            "model": model_rows[0]["model"],
            "stance": "mean",
            "difference": sum(row["difference"] for row in model_rows) / 2,
        })
        rows.extend(model_rows)

    results = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
