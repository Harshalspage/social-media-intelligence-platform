import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sentiment_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "reports" / "evaluation" / "sentiment_trends.csv"
)


def run_sentiment_trends():

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("No sentiment records found.")

    counts = Counter()
    totals = Counter()

    for record in records:

        try:
            timestamp = datetime.fromisoformat(
                record["created_at"].replace("Z", "+00:00")
            )
        except (KeyError, ValueError, TypeError):
            continue

        platform = record.get("platform", "").lower()
        sentiment = record.get("sentiment_label", "").lower()

        if sentiment not in {"positive", "neutral", "negative"}:
            continue

        date = timestamp.date().isoformat()

        counts[(date, platform, sentiment)] += 1
        totals[(date, platform)] += 1

    rows = []

    for (date, platform, sentiment), count in sorted(counts.items()):

        total = totals[(date, platform)]

        rows.append({
            "date": date,
            "platform": platform,
            "sentiment_label": sentiment,
            "post_count": count,
            "platform_total": total,
            "sentiment_percentage": round(
                count / total * 100, 2
            )
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "date",
                "platform",
                "sentiment_label",
                "post_count",
                "platform_total",
                "sentiment_percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Sentiment trend analysis completed.")
    print(f"Records loaded: {len(records)}")
    print(f"Summary rows: {len(rows)}")
    print(f"Report saved to: {OUTPUT_FILE}")

    for row in rows:
        print(row)


if __name__ == "__main__":
    run_sentiment_trends()