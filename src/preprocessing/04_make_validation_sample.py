"""Create a deterministic validation import for independent annotators."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "main_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "labelstudio_batches" / "validation"
SEED = 20261105
MAIN_COUNTS = {
    "pro_allgemein": 35,
    "pro_massnahme": 40,
    "contra_allgemein": 35,
    "contra_massnahme": 40,
    "neutral": 50,
}
def select_stratified(frame, counts, rng):
    selected = []
    for label, count in counts.items():
        candidates = frame[frame["stance"].map(str).str.strip().str.casefold() == label].copy()
        candidates = candidates.sort_values("labelstudio_task_id")
        candidates = candidates.iloc[rng.permutation(len(candidates))]
        projects = list(candidates["labelstudio_project"].fillna("unknown").unique())
        groups = {
            project: candidates[candidates["labelstudio_project"].fillna("unknown").eq(project)]
            for project in projects
        }
        picked = []
        while len(picked) < count:
            progressed = False
            for project in projects:
                if len(groups[project]):
                    picked.append(groups[project].iloc[0])
                    groups[project] = groups[project].iloc[1:]
                    progressed = True
                    if len(picked) == count:
                        break
            if not progressed:
                raise ValueError(f"Not enough rows for stance {label}: need {count}")
        selected.extend(picked)
    return pd.DataFrame(selected)


def write_import(frame, path):
    rows = [
        {
            "data": {
                "text": row.text,
                "code": f"V{index:03d}",
                "candidate_id": row.candidate_id,
            }
        }
        for index, row in enumerate(frame.itertuples(index=False), start=1)
    ]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return [row["data"]["code"] for row in rows]


def main():
    rng = np.random.default_rng(SEED)
    frame = pd.read_csv(INPUT).sort_values("labelstudio_task_id").reset_index(drop=True)
    main = select_stratified(frame, MAIN_COUNTS, rng)
    main = main.iloc[rng.permutation(len(main))].reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_codes = write_import(main, OUTPUT_DIR / "import_main.json")
    key_rows = []
    for code, row in zip(main_codes, main.itertuples(index=False)):
        key_rows.append(
            {
                "code": code,
                "id": row.labelstudio_task_id,
                "project": row.labelstudio_project,
                "stance": row.stance,
                "stance_3": row.stance_3,
                "tone": row.tone,
                "confidence": row.confidence,
                "party": row.party,
            }
        )
    pd.DataFrame(key_rows).to_csv(OUTPUT_DIR / "PRIVATE_key.csv", index=False)
    print("validation distribution:")
    print(pd.crosstab(main["stance"], main["labelstudio_project"]).to_string())
    print(f"Wrote {len(main):,} validation items.")


if __name__ == "__main__":
    main()
