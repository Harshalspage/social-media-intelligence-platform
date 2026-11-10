
import json
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "clean_posts.json"
)

RESULT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "topic_model_results.json"
)

KEYWORD_FILE = (
    PROJECT_ROOT / "reports" / "evaluation" / "topic_keywords.csv"
)


def run_topic_modeling():

    print("Starting topic modeling...")

    # 1. Load processed posts
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("No posts found in the input dataset.")

    # 2. Extract text
    valid_records = [
        record for record in records
        if record.get("text", "").strip()
    ]

    documents = [record["text"] for record in valid_records]

    if len(documents) < 2:
        raise ValueError("At least two valid posts are required.")

    # 3. Convert text into TF-IDF features
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # 4. Determine number of topics
    n_topics = min(3, tfidf_matrix.shape[0], tfidf_matrix.shape[1])

    if n_topics < 2:
        raise ValueError("Not enough vocabulary to discover topics.")

    # 5. Train NMF topic model
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        init="nndsvda",
        max_iter=500
    )

    document_topics = nmf_model.fit_transform(tfidf_matrix)

    # 6. Extract topic keywords
    feature_names = vectorizer.get_feature_names_out()

    topic_keywords = []

    for topic_id, topic_weights in enumerate(nmf_model.components_):

        top_indices = topic_weights.argsort()[::-1][:8]

        keywords = [
            feature_names[index]
            for index in top_indices
            if topic_weights[index] > 0
        ]

        topic_keywords.append({
            "topic_id": int(topic_id),
            "keywords": keywords
        })

    # 7. Assign topics to posts
    results = []

    for index, record in enumerate(valid_records):

        topic_scores = document_topics[index]

        topic_id = int(topic_scores.argmax())

        total_score = topic_scores.sum()

        topic_strength = (
            float(topic_scores[topic_id] / total_score)
            if total_score > 0 else 0.0
        )

        results.append({
            **record,
            "topic_id": topic_id,
            "topic_keywords": topic_keywords[topic_id]["keywords"],
            "topic_strength": round(topic_strength, 4)
        })

    # 8. Save post-level topic results
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    # 9. Save topic keywords
    KEYWORD_FILE.parent.mkdir(parents=True, exist_ok=True)

    with KEYWORD_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["topic_id", "keywords"]
        )

        writer.writeheader()

        for topic in topic_keywords:
            writer.writerow({
                "topic_id": topic["topic_id"],
                "keywords": ", ".join(topic["keywords"])
            })

    # 10. Display results
    print("\nTOPIC MODELING COMPLETED")
    print("------------------------")
    print(f"Posts analysed: {len(results)}")
    print(f"Topics discovered: {n_topics}")

    print("\nDISCOVERED TOPICS:")

    for topic in topic_keywords:
        print(
            f"\nTopic {topic['topic_id']}: "
            f"{', '.join(topic['keywords'])}"
        )

    print(f"\nPost results saved: {RESULT_FILE}")
    print(f"Topic keywords saved: {KEYWORD_FILE}")


if __name__ == "__main__":
    run_topic_modeling()
