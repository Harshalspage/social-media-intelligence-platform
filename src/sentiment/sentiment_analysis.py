
import json
from pathlib import Path

from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "clean_posts.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sentiment_results.json"
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def run_sentiment_analysis():
    # Load the processed posts from our ingestion pipeline.
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        posts = json.load(file)

    if not posts:
        raise ValueError("No posts found in the processed dataset.")

    # Load the pretrained Hugging Face sentiment model.
    print(f"Loading sentiment model: {MODEL_NAME}")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,  # Run on CPU
    )

    texts = [post["text"] for post in posts]

    # Predict sentiment for each post.
    predictions = classifier(
        texts,
        batch_size=4,
        truncation=True,
        max_length=512,
    )

    results = []

    for post, prediction in zip(posts, predictions):
        result = post.copy()
        result["sentiment_label"] = prediction["label"].lower()
        result["sentiment_score"] = round(
            float(prediction["score"]), 4
        )
        result["sentiment_model"] = MODEL_NAME
        results.append(result)

    # Save the enriched dataset.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print("\nSentiment analysis completed.")
    print(f"Total posts analyzed: {len(results)}")
    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nPredictions:")
    for result in results:
        print(
            f"[{result['sentiment_label'].upper():8}] "
            f"Score: {result['sentiment_score']:.4f} | "
            f"{result['text']}"
        )


if __name__ == "__main__":
    run_sentiment_analysis()
