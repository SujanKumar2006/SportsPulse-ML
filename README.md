# ⚽🏏 SportsPulse ML

> A dual-sport machine learning analytics system covering **English Premier League football** and **Indian Premier League cricket** — built as an academic AAT project.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What This Project Does

SportsPulse ML is a complete end-to-end machine learning pipeline that downloads real sports data, trains multiple ML models, generates dynamic charts, and serves everything through an interactive web dashboard — all from Python scripts you can run on any machine.

---

## 🗂️ Project Structure

```
SportsPulse-ML/
│
├── football/
│   ├── sports_ml_real.py      ← Core ML pipeline (runs headless, saves charts)
│   └── football_app.py        ← Streamlit interactive web app
│
├── ipl/
│   └── ipl_cricket_ml.py      ← Full IPL ML system (5 modules + browser viewer)
│
├── requirements.txt           ← All dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/SportsPulse-ML.git
cd SportsPulse-ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run what you want

| What | Command |
|------|---------|
| Football charts (headless) | `python football/sports_ml_real.py` |
| Football web app | `cd football && streamlit run football_app.py` |
| IPL analytics + browser viewer | `python ipl/ipl_cricket_ml.py` |

---

## ⚽ Module 1 — Football Analytics (`football/`)

### `sports_ml_real.py` — Core Pipeline

Downloads real FIFA player stats and EPL match results, trains 4 ML models, and saves 7 charts + a self-contained HTML dashboard to an `outputs/` folder.

**Data sources (auto-downloaded):**
- FIFA 20/21 player stats — from public GitHub mirrors
- EPL match results — from [football-data.co.uk](https://www.football-data.co.uk)

**4 ML Models trained:**

| Module | Algorithm | Task |
|--------|-----------|------|
| A | Random Forest / Gradient Boosting | Match outcome prediction (H/D/A) |
| B | Linear / Ridge Regression | Player performance scoring |
| C | K-Means Clustering (k=3/4/5) | Player archetype discovery |
| D | Cosine Similarity | Similar player recommendation |

**7 output charts saved:**
1. `1_eda_analysis.png` — Exploratory data analysis (6-panel)
2. `2_match_predictor.png` — Confusion matrix + feature importances
3. `3_performance_scorer.png` — Actual vs predicted + coefficients
4. `4_player_clusters.png` — Elbow method + cluster scatter
5. `5_player_recommender.png` — Similarity heatmap + radar chart
6. `6_team_form.png` — League table + win/draw/loss breakdown
7. `7_master_dashboard.png` — Dark-themed KPI dashboard

> 💡 **Dynamic seed** — every run produces different train/test splits, chart colours, model variants and recommendations. No two runs look the same.

---

### `football_app.py` — Streamlit Web App

An interactive 7-page web dashboard powered by the same ML models.

```bash
cd football
streamlit run football_app.py
```

Opens at **http://localhost:8501**

**7 pages:**

| Page | Description |
|------|-------------|
| 🏠 Dashboard | Live KPIs, top players chart, results distribution |
| 🏆 Match Predictor | Drag sliders for match stats → instant Win/Draw/Loss prediction with probabilities |
| 📈 Performance Scorer | Dial in any player's attributes → predicted score + percentile rank |
| 🗂️ Player Clusters | Explore 4 player archetypes with interactive scatter (any X/Y axis) |
| 🤖 Player Recommender | Search any player → radar comparison + similar player cards |
| 🌍 EDA Explorer | Filter by position/age/rating, custom scatter plots, team-level match stats |
| 📊 League Table | Full standings, goals chart, head-to-head lookup |

---

## 🏏 Module 2 — IPL Cricket Analytics (`ipl/`)

### `ipl_cricket_ml.py` — 5-Module IPL System

A complete IPL analytics pipeline covering match prediction, player scoring, best XI selection, batting/bowling phase analysis, and a live score tracker.

**Data strategy (3-tier fallback):**
```
Tier 1 → Cricsheet official zip   (auto-downloads, no login needed)
Tier 2 → Kaggle CLI               (auto-downloads with one-time setup)
Tier 3 → Local CSV files          (manual drop-in)
       → Rich generated data      (always works as fallback)
```

> Once real data is downloaded it is cached locally — no re-downloading on future runs.

**5 ML Modules:**

| Module | Algorithm | Task |
|--------|-----------|------|
| A | Random Forest | IPL match winner prediction |
| B | Ridge Regression | Player performance scoring |
| C | Cosine Similarity | Best Playing XI recommender |
| D | Analytics | Batting/bowling phase analysis (6 charts) |
| E | API integration | Live score tracker (Cricbuzz via RapidAPI) |

**6 output charts + HTML dashboard** saved to `ipl_outputs/` — auto-opens in browser.

#### Enable real live scores (Module E)

```python
# In ipl_cricket_ml.py, replace line ~60:
RAPID_KEY = "YOUR_RAPIDAPI_KEY"
# with your free key from rapidapi.com/cricbuzz (500 calls/month, no credit card)
```

#### Get real IPL data (Cricsheet — easiest, no login)

```
1. Go to  https://cricsheet.org/downloads/
2. Click  "IPL" under CSV downloads
3. Extract the zip → paste all CSV files into:
   ipl/ipl_data_cache/
4. Re-run the script — auto-detected!
```

---

## 📦 Dependencies

```
pandas          — data manipulation
numpy           — numerical computing
scikit-learn    — all ML models
matplotlib      — chart generation
seaborn         — statistical plots
requests        — data downloading
streamlit       — web app framework
```

Install everything at once:
```bash
pip install -r requirements.txt
```

---

## 🧠 ML Concepts Demonstrated

- **Supervised learning** — classification (match outcome) and regression (player performance)
- **Unsupervised learning** — K-Means clustering for player archetypes
- **Similarity search** — cosine similarity for player recommendations
- **Feature engineering** — encoding categorical variables, normalisation, scaling
- **Model evaluation** — accuracy, cross-validation, R², RMSE, confusion matrix
- **Exploratory Data Analysis** — correlation heatmaps, distribution plots, scatter analysis

---

## 🖼️ Sample Output

The pipeline generates a dark-themed master dashboard that summarises all model results in a single view — including KPI cards, feature importance charts, cluster visualisations, and league standings.

Charts are saved to:
- `football/outputs/` — football pipeline
- `ipl/ipl_outputs/` — IPL pipeline

---

## 📖 Academic Context

This project was built as part of an **AAT (Applied Academic Technology) coursework** to demonstrate practical machine learning skills using real-world sports data. It covers the full ML workflow from raw data ingestion through to interactive deployment.

---

## 🤝 Contributing

Pull requests are welcome! If you have ideas for new modules (xG model, player injury predictor, transfer value estimator, fantasy XI optimiser) feel free to open an issue.

---

## 📄 License

MIT License — free to use, modify and distribute with attribution.

---

<p align="center">Built with Python · scikit-learn · Streamlit · Matplotlib</p>
