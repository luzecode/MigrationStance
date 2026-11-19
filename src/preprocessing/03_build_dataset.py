"""Merge, clean, and build datasets from the latest Label Studio exports."""

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "data" / "labelstudio_batches"
EXPORT_DIR = BATCH_DIR / "export"
OUTPUT_DIR = ROOT / "data" / "processed"
RAW_BATCH_DIR = ROOT / "data" / "raw" / "batches"


def read_candidates():
    files = sorted(RAW_BATCH_DIR.glob("batch_*.csv"))
    if not files:
        raise FileNotFoundError(f"No candidate batches found in {RAW_BATCH_DIR}")
    tables = [pd.read_csv(file) for file in files]
    candidates = pd.concat(tables, ignore_index=True)
    return candidates.drop_duplicates("candidate_id", keep="last")


def project_number(path):
    return int(path.name.split("-at-", 1)[0].removeprefix("project-"))


def latest_exports():
    files = [
        path
        for path in EXPORT_DIR.iterdir()
        if path.is_file()
        and path.name.startswith("project-")
        and not path.name.endswith("-info.json")
        and path.suffix.lower() in {".csv", ".json"}
    ]
    latest = {}
    for path in files:
        project = project_number(path)
        if project not in latest or path.stat().st_mtime > latest[project].stat().st_mtime:
            latest[project] = path
    return [latest[number] for number in sorted(latest)]


def add_result(row, result):
    name = result.get("from_name")
    value = result.get("value", {})
    if not name:
        return
    if "choices" in value:
        row[name] = " | ".join(map(str, value["choices"]))
    elif "text" in value:
        row[name] = " ".join(map(str, value["text"]))


def read_json_export(path):
    tasks = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        row = dict(task.get("data", {}))
        row["labelstudio_project"] = task.get("project")
        row["labelstudio_task_id"] = task.get("id")
        annotations = task.get("annotations", [])
        if annotations:
            for result in annotations[0].get("result", []):
                add_result(row, result)
        rows.append(row)
    return rows


def read_csv_export(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))
    rows = []
    for source_row in source_rows:
        row = dict(source_row)
        row["labelstudio_project"] = row.get("project") or row.get("labelstudio_project")
        row["labelstudio_task_id"] = (
            row.get("id") or row.get("task_id") or row.get("labelstudio_task_id")
        )
        annotation_value = row.get("annotations") or row.get("annotation")
        if annotation_value:
            try:
                annotations = json.loads(annotation_value)
            except json.JSONDecodeError:
                annotations = []
            if isinstance(annotations, list) and annotations:
                for result in annotations[0].get("result", []):
                    add_result(row, result)
        rows.append(row)
    return rows


def read_export(path):
    if path.suffix.lower() == ".json":
        return read_json_export(path)
    return read_csv_export(path)


def normalise(value):
    if pd.isna(value):
        return ""
    return str(value).strip().casefold().replace(" ", "_")


def clean_table(candidates, annotations):
    annotations = pd.DataFrame(annotations).drop_duplicates(
        "labelstudio_task_id", keep="first"
    )
    merged = candidates.merge(
        annotations,
        on="candidate_id",
        how="left",
        suffixes=("", "_annotation"),
    )
    for annotation_column in [
        column for column in merged.columns if column.endswith("_annotation")
    ]:
        column = annotation_column.removesuffix("_annotation")
        if column in merged:
            merged[column] = merged[annotation_column].combine_first(merged[column])
            merged = merged.drop(columns=annotation_column)

    clean = merged.copy()
    clean.columns = clean.columns.str.strip()
    for column in clean.select_dtypes(include=["object", "string"]).columns:
        clean[column] = clean[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    clean = clean.replace(r"^\s*$", pd.NA, regex=True)
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    for column in ["period", "session", "chunk_idx", "n_sentences", "n_words"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").astype("Int64")
    clean = clean.dropna(subset=["candidate_id", "text"])
    clean = clean.drop_duplicates("candidate_id", keep="last")
    clean = clean.sort_values(["period", "session", "chunk_idx"])
    clean["stance_3"] = normalise_series(clean["stance"]).map(
        {
            "pro_allgemein": "pro",
            "pro_massnahme": "pro",
            "contra_allgemein": "contra",
            "contra_massnahme": "contra",
            "neutral": "neutral",
        }
    )
    return clean


def normalise_series(series):
    return series.astype("string").str.strip().str.casefold().str.replace(" ", "_")


def print_summary(clean):
    relevant = clean[normalise_series(clean["topic_relevance"]).eq("relevant")]
    stance_counts = relevant["stance_3"].value_counts().to_dict()
    tone_counts = normalise_series(relevant["tone"]).value_counts().to_dict()
    print(f"Unique tasks: {clean['labelstudio_task_id'].nunique():,}")
    print(f"Relevant rows: {len(relevant):,}")
    print(f"Stance 3: {stance_counts}")
    print(f"Tone: {tone_counts}")


def main():
    exports = latest_exports()
    if not exports:
        raise FileNotFoundError(f"No Label Studio exports found in {EXPORT_DIR}")
    candidates = read_candidates()
    annotations = [row for export in exports for row in read_export(export)]
    clean = clean_table(candidates, annotations)
    print_summary(clean)
    relevant = clean[
        normalise_series(clean["topic_relevance"]).eq("relevant")
        & clean["stance_3"].notna()
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_DIR / "merged_cleaned_annotations.csv", index=False)
    relevant.to_csv(OUTPUT_DIR / "main_dataset.csv", index=False)
    print(f"Wrote {len(clean):,} merged rows.")
    print(f"Wrote {len(relevant):,} main-dataset rows.")


if __name__ == "__main__":
    main()
