"""Prepare the labelled corpus for German stance NLI fine-tuning."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "main_dataset.csv"
VALIDATION_KEY = ROOT / "data" / "labelstudio_batches" / "validation" / "PRIVATE_key.csv"
OUTPUT_DIR = ROOT / "data" / "nli"
HYPOTHESIS = "Migration nach Deutschland soll reduziert oder restriktiver gehandhabt werden."
SEED = 20261105

STANCE_TO_NLI = {
    "pro": "entailment",
    "neutral": "neutral",
    "contra": "contradiction",
}


def main():
    data = pd.read_csv(INPUT)
    if "stance_3" not in data.columns:
        raise KeyError("main_dataset.csv must contain a stance_3 column")

    data = data[data["stance_3"].isin(STANCE_TO_NLI)].copy()
    data["hypothesis"] = HYPOTHESIS
    data["nli_label"] = data["stance_3"].map(STANCE_TO_NLI)

    if VALIDATION_KEY.exists():
        validation = pd.read_csv(VALIDATION_KEY, dtype=str)
        validation_ids = set(validation["id"].dropna().astype(str))
        data["is_validation"] = data["labelstudio_task_id"].astype(str).isin(validation_ids)
    else:
        data["is_validation"] = False

    columns = [
        "candidate_id",
        "text",
        "hypothesis",
        "nli_label",
        "stance_3",
        "tone",
        "labelstudio_project",
        "labelstudio_task_id",
        "period",
        "party",
        "confidence",
        "is_validation",
    ]
    dataset = data[[column for column in columns if column in data.columns]].copy()
    dataset = dataset.rename(columns={"text": "premise"})

    train_pool = dataset[~dataset["is_validation"]].copy()
    test = dataset[dataset["is_validation"]].copy()

    train, dev = train_test_split(
        train_pool,
        test_size=0.10,
        random_state=SEED,
        stratify=train_pool["nli_label"],
    )
    train = train.copy()
    dev = dev.copy()
    train["split"] = "train"
    dev["split"] = "dev"
    test["split"] = "test"

    prepared = pd.concat([train, dev, test], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT_DIR / "nli_dataset.csv", index=False)
    train.to_csv(OUTPUT_DIR / "nli_train.csv", index=False)
    dev.to_csv(OUTPUT_DIR / "nli_dev.csv", index=False)
    test.to_csv(OUTPUT_DIR / "nli_test.csv", index=False)

    metadata = {
        "hypothesis": HYPOTHESIS,
        "label_mapping": STANCE_TO_NLI,
        "baseline_model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "seed": SEED,
        "counts": {
            "all": len(prepared),
            "train": len(train),
            "dev": len(dev),
            "test_validation": len(test),
        },
        "label_counts": prepared["nli_label"].value_counts().to_dict(),
        "split_label_counts": {
            split: frame["nli_label"].value_counts().to_dict()
            for split, frame in prepared.groupby("split")
        },
    }
    (OUTPUT_DIR / "nli_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
