# Large-Scale Social Media Intelligence Platform

An end-to-end Data Science project designed to analyse publicly accessible social media discussions from Reddit and YouTube using Natural Language Processing, Machine Learning and Graph Analytics.

## Project Overview

The platform aims to transform unstructured social media conversations into meaningful insights by analysing sentiment, discovering discussion topics, identifying interaction patterns and detecting potentially misleading information.

## Project Objectives

- Collect publicly accessible social media data from Reddit and YouTube through approved APIs.
- Clean and preprocess unstructured text.
- Perform sentiment classification and sentiment trend analysis.
- Discover discussion topics using topic modelling.
- Analyse user interaction networks and identify influential nodes.
- Develop a machine learning model for misinformation classification.
- Provide searchable data and visual analytics through Elasticsearch and Kibana.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| NLP | Transformers, NLTK |
| Sentiment Analysis | CardiffNLP RoBERTa |
| Topic Modelling | TF-IDF, NMF, Scikit-learn |
| Machine Learning | Scikit-learn |
| Graph Analytics | NetworkX, Neo4j |
| Search | Elasticsearch |
| Visualization | Kibana, Plotly |
| Version Control | Git, GitHub |

## Project Structure

```text
social-media-intelligence-platform/
│
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── dashboards/
│   └── kibana/
│
├── docs/
├── reports/
│   ├── evaluation/
│   └── figures/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── sentiment/
│   ├── topic_modeling/
│   ├── graph_analytics/
│   ├── misinformation/
│   └── search/
│
├── tests/
├── .env.example
├── .gitignore
└── README.md
```

## Implemented Features

### 1. Data Ingestion
- Initial sample data ingestion using JSON.
- Basic record validation and duplicate handling.
- Storage of raw and processed records.

### 2. Text Preprocessing
- Initial text cleaning and data validation.
- Structured data preparation for downstream analysis.

### 3. Sentiment Analysis
- Integrated CardiffNLP Twitter RoBERTa sentiment model.
- Classified sample posts into positive and negative sentiment categories.
- Generated sentiment results with model scores.

### 4. Sentiment Trend Analysis
- Aggregated sentiment by platform and date.
- Exported sentiment summaries to CSV.

### 5. Topic Modeling
- Implemented TF-IDF vectorization and Non-negative Matrix Factorization (NMF).
- Extracted topic keywords.
- Assigned topic IDs to individual posts.
- Exported topic results in JSON and CSV formats.

## Planned Features

- Reddit and YouTube API integration.
- Expanded multilingual sentiment analysis.
- Social network analysis using NetworkX and Neo4j.
- Misinformation classification using labelled datasets.
- Elasticsearch integration and Kibana dashboards.
- Integrated analytics dashboard and final evaluation.

## Current Development Status

This project is under active development. Initial analytical modules have been implemented and tested using a small sample dataset of five posts. Live API integration and the remaining analytical modules are pending.

## Project Information

- **Project Type:** Individual Data Science Capstone Project
- **Development Start:** 20 October 2026
- **Submission Deadline:** 20 November 2026
- **Status:** In Progress

## Data and Responsible Use

The project is intended for educational and research purposes. External platform data will only be collected subject to applicable API approval, platform terms, privacy requirements and usage restrictions.

No API credentials or private user information should be committed to this repository.

## Author

Harshal Somnath Shirsat