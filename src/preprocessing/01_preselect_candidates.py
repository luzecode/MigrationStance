"""Create an unlabelled migration candidate set from PoliCorp periods 18-20."""

import argparse
import csv
import json
import re
from pathlib import Path


TERMS = (
    r"asyl\w*|flücht\w*|geflüchtet\w*|migration\w*|migrant\w*|"
    r"zuwander\w*|einwander\w*|abschieb\w*|rückführ\w*|schutzsuchend\w*|"
    r"aufenthalt\w*|familiennachzug|subsidiär\w*|herkunftsstaat\w*|"
    r"grenzkontroll\w*|zurückweis\w*|integrationskurs\w*|geas|"
    r"ausreisepflicht\w*|duldung\w*|obergrenze"
)
TERM_RE = re.compile(TERMS, re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ§])")
WORD_RE = re.compile(r"\b[\wÄÖÜäöüß-]+\b")

POSITIVE = re.compile(
    r"\b(chance\w*|bereicher\w*|willkommen\w*|erfolgreich\w*|solidar\w*|"
    r"menschlich\w*|integration\w*|schutz\w*|hilf\w*)\b", re.IGNORECASE
)
NEGATIVE = re.compile(
    r"\b(krise\w*|problem\w*|belast\w*|illegal\w*|grenz\w*|sicher\w*|"
    r"gefähr\w*|abschieb\w*|überforder\w*|missbrauch\w*|terror\w*)\b", re.IGNORECASE
)
PRO_LIMIT = re.compile(
    r"\b(begrenz\w*|limitier\w*|obergrenze\w*|reduzier\w*|weniger\w*|"
    r"stoppen?\w*|stopp\w*|zurückweis\w*|abschieb\w*)\b", re.IGNORECASE
)
CONTRA_LIMIT = re.compile(
    r"\b(keine obergrenze|nicht begrenz\w*|mehr schutz|aufnehmen\w*|"
    r"willkommen\w*|menschenrecht\w*|asylrecht\w*)\b", re.IGNORECASE
)


def words(text):
    return len(WORD_RE.findall(text))


def flatten_text(segments):
    """Keep speech segments and discard structured interjections."""
    return " ".join(
        segment.get("text", "").strip()
        for segment in segments
        if segment.get("type") in {"speech", "call to order", "call for order"}
    )


def clean(text):
    text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"(?i)\b([A-ZÄÖÜ][\wÄÖÜäöüß]*)-\s+(?=[a-zäöüß])", r"\1", text)
    text = re.sub(r"\bCO\s+2\b", "CO2", text)
    return re.sub(r"\s+", " ", text).strip()


def sentences(text):
    protected = re.sub(
        r"\b(Dr|Abs|Art|Nr|z|u|bzw|ca)\.\s*", r"\1§", text, flags=re.IGNORECASE
    )
    protected = re.sub(r"(?<=\d)\.(?=\s*(?:Lesung|Januar|Februar|März|April|Mai|Juni|"
                       r"Juli|August|September|Oktober|November|Dezember))", "§", protected)
    return [part.replace("§", ".").strip() for part in SENTENCE_RE.split(protected) if part.strip()]


def chunks(text):
    result = []
    current = []
    count = 0
    for sentence in sentences(text):
        n = words(sentence)
        if n > 150:
            if current:
                result.append((" ".join(current), False))
                current, count = [], 0
            result.append((sentence, True))
        elif current and count + n > 150:
            if len(current) >= 2 and count >= 40:
                result.append((" ".join(current), False))
            current, count = [sentence], n
        else:
            current.append(sentence)
            count += n
            if len(current) == 4 or (len(current) >= 2 and count >= 40):
                result.append((" ".join(current), False))
                current, count = [], 0
    if current and len(current) >= 2 and count >= 40:
        result.append((" ".join(current), False))
    return result


def scores(text):
    positive = len(POSITIVE.findall(text))
    negative = len(NEGATIVE.findall(text))
    stance = len(PRO_LIMIT.findall(text)) - len(CONTRA_LIMIT.findall(text))
    return positive - negative, stance


def main():
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=project_dir / "data/raw/germaparl2-plus_lp_subsets"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data/processed/candidates_migration_18_20.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for period in (18, 19, 20):
        source = args.data_dir / f"policorp_lp-{period}.json"
        with source.open(encoding="utf-8") as file:
            records = json.load(file)["data"]
        for record in records:
            if record.get("role") == "presidency":
                continue
            topic = next(
                (item for item in record.get("topic", []) if item.get("label") == "Immigration"),
                None,
            )
            raw = clean(flatten_text(record.get("text_raw", [])))
            if not topic and not TERM_RE.search(raw):
                continue
            speech_chunks = chunks(raw)
            for chunk_index, (text, long_sentence) in enumerate(speech_chunks):
                matches = sorted({match.lower() for match in TERM_RE.findall(text)})
                if not matches or words(text) < 30:
                    continue
                sentiment_score, stance_score = scores(text)
                rows.append(
                    {
                        "candidate_id": f"{period}_{record['speech_id']}_{chunk_index}",
                        "speech_id": record["speech_id"],
                        "period": period,
                        "session": record["sessionno"],
                        "date": record["date"],
                        "agenda_id": record["agenda_id"],
                        "agenda_type": record.get("agenda_type"),
                        "speaker": record.get("name"),
                        "party": record.get("party"),
                        "role": record.get("role"),
                        "topic_label": "Immigration" if topic else "",
                        "topic_score": topic.get("score") if topic else "",
                        "chunk_idx": chunk_index,
                        "n_sentences": len(sentences(text)),
                        "n_words": words(text),
                        "long_sentence": long_sentence,
                        "migration_terms": "|".join(matches),
                        "sentiment_score": sentiment_score,
                        "stance_score": stance_score,
                        "text": text,
                        "stance": "",
                        "tone": "",
                    }
                )

    fields = list(rows[0]) if rows else []
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} candidates to {args.output}")


if __name__ == "__main__":
    main()
