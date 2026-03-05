# Distributed Hybrid Recommender System

A production-grade recommendation engine built with **Apache Spark MLlib**, combining Collaborative Filtering (ALS) and Content-Based Filtering using local sentence embeddings — no API key required.

---

## How It Works

The system uses two engines blended into a single score:

### 1. Collaborative Filtering — ALS (70% weight)
Learns from user behaviour. The user-item rating matrix is factorised into hidden "taste" vectors using Alternating Least Squares. Users with similar vectors get similar recommendations. Runs fully distributed across Spark partitions — scales to billions of interactions.

### 2. Content-Based Filtering — Sentence Embeddings (30% weight)
Learns from product content. Each product's title + description is encoded into a 384-dimensional semantic vector using `all-MiniLM-L6-v2`. Products are clustered via KMeans — items in the same cluster are semantically related and serve as fallback candidates.

### 3. Hybrid Fusion
```
final_score = 0.7 × ALS_score + 0.3 × content_score
```
Active users get personalised results. Cold-start users (no history) still get sensible content-matched suggestions.

---

## Project Structure

```
├── recommender_system.ipynb   # Main notebook (run in Google Colab)
├── products_dataset.csv       # Product catalog (upload to Colab)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to exclude from git
└── README.md                  # This file
```

---

## Quick Start

### Run in Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `recommender_system.ipynb`
3. Upload `products_dataset.csv` to the Colab file browser
4. Run all cells — no API key needed

### Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/hybrid-recommender-system.git
cd hybrid-recommender-system
pip install -r requirements.txt
jupyter notebook recommender_system.ipynb
```

> **Note:** Running locally requires Java 8+ installed for PySpark.  
> Check with `java -version`. Install from [adoptium.net](https://adoptium.net/) if needed.

---

## Tech Stack

| Component | Technology |
|---|---|
| Distributed computing | Apache Spark (PySpark) |
| Collaborative Filtering | MLlib ALS |
| Hyperparameter tuning | MLlib CrossValidator + ParamGridBuilder |
| Text embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Clustering | MLlib KMeans |
| Dimensionality reduction | MLlib PCA |
| Visualisation | Plotly Express |

---

## Pipeline Overview

```
Raw Data
   │
   ├─► Synthetic user-item interactions (500 users × 2000 products)
   │
   ├─► [ALS Branch]
   │       Train/test split (80/20)
   │       Baseline ALS → RMSE / MAE evaluation
   │       CrossValidator hyperparameter tuning
   │       Top-N recommendations per user
   │
   ├─► [Content Branch]
   │       title + description → sentence-transformers (384-dim)
   │       Elbow method → optimal k
   │       KMeans clustering (k=8)
   │       PCA → 2D visualisation
   │
   └─► Hybrid Fusion → ranked Top-10 recommendations
```

---

## Key Results

- **RMSE** evaluated on held-out 20% test set
- **Hyperparameter tuning** over `rank` ∈ {20, 50} and `regParam` ∈ {0.01, 0.1, 0.5}
- **8 semantic product clusters** visualised in 2D via PCA
- Cold-start users handled gracefully via content fallback

---

## Configuration

You can tune the following constants at the top of the notebook:

| Variable | Default | Description |
|---|---|---|
| `NUM_USERS` | 500 | Simulated user base size |
| `NUM_CLUSTERS` | 8 | KMeans cluster count |
| `TOP_N` | 10 | Number of recommendations to return |
| `ALPHA` | 0.7 | ALS weight in hybrid fusion |

---

## Production Extensions

- **Real-time serving** — export ALS latent factors to Redis/Feast, serve via FastAPI
- **Implicit feedback** — set `implicitPrefs=True` to use clicks/views as confidence weights
- **Delta Lake** — ACID-compliant streaming ingestion of interaction logs
- **MLflow** — `mlflow.spark.autolog()` for full experiment tracking
- **A/B testing** — shadow-deploy new models and compare CTR/CVR online

---

## License

MIT License — free to use, modify, and distribute.
