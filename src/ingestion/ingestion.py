
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FILE = PROJECT_ROOT / "data" / "sample" / "sample_posts.json"
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "ingested_posts.json"
PROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "clean_posts.json"
)

REQUIRED_FIELDS = {
    "post_id",
    "platform",
    "text",
    "created_at",
    "author",
    "community",
}


def load_json(file_path):
    """Read a JSON file containing a list of social media records."""
    with file_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of records.")

    return records


def validate_record(record):
    """Check required fields and basic data quality."""
    if not isinstance(record, dict):
        return False

    if not REQUIRED_FIELDS.issubset(record.keys()):
        return False

    if not isinstance(record["post_id"], str) or not record["post_id"].strip():
        return False

    if record["platform"] not in {"reddit", "youtube"}:
        return False

    if not isinstance(record["text"], str) or not record["text"].strip():
        return False

    return True


def save_json(records, file_path):
    """Save records as readable UTF-8 JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def run_ingestion():
    # Load our initial sample dataset.
    records = load_json(SAMPLE_FILE)

    # Preserve the original records before processing.
    save_json(records, RAW_FILE)

    valid_records = []
    seen_ids = set()

    for record in records:
        if not validate_record(record):
            continue

        post_id = record["post_id"].strip()

        if post_id in seen_ids:
            continue

        seen_ids.add(post_id)

        cleaned_record = record.copy()
        cleaned_record["post_id"] = post_id
        cleaned_record["text"] = " ".join(
            cleaned_record["text"].split()
        )
        cleaned_record["platform"] = (
            cleaned_record["platform"].lower()
        )

        valid_records.append(cleaned_record)

    save_json(valid_records, PROCESSED_FILE)

    print("Ingestion completed.")
    print(f"Records loaded: {len(records)}")
    print(f"Valid unique records: {len(valid_records)}")
    print(f"Rejected or duplicate records: {len(records) - len(valid_records)}")
    print(f"Raw data saved to: {RAW_FILE}")
    print(f"Processed data saved to: {PROCESSED_FILE}")


if __name__ == "__main__":
    run_ingestion()
