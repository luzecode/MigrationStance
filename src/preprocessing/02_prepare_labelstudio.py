"""Prepare the migration candidates for Label Studio."""

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processed" / "candidates_migration_18_20.csv"
OUTPUT = ROOT / "data" / "processed" / "annotation_candidates.csv"

FIELDS = [
    "candidate_id",
    "speech_id",
    "period",
    "session",
    "date",
    "agenda_id",
    "speaker",
    "party",
    "role",
    "chunk_idx",
    "n_sentences",
    "n_words",
    "long_sentence",
    "migration_terms",
    "text",
    "topic_relevance",
    "stance",
    "tone",
    "annotator_notes",
]

def readable_id(row):
    speech_number = row["speech_id"].rsplit("_", 1)[-1]
    source_key = re.sub(r"[^A-Za-z0-9]+", "", row["speech_id"])
    return (
        f"P{int(row['period']):02d}_S{int(row['session']):03d}_"
        f"SP{int(speech_number):03d}_C{int(row['chunk_idx']):03d}_SRC{source_key}"
    )


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            clean_row = {field: row.get(field, "") for field in FIELDS}
            clean_row["candidate_id"] = readable_id(row)
            clean_row["topic_relevance"] = ""
            clean_row["stance"] = ""
            clean_row["tone"] = ""
            clean_row["annotator_notes"] = ""
            writer.writerow(clean_row)

    print(f"Wrote {len(rows):,} annotation rows to {OUTPUT}")


if __name__ == "__main__":
    main()
