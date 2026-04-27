# CS301 Avocado ML Project

This project implements the full CS301 3-stage workflow using the Kaggle avocado prices dataset:
- Stage 1: data cleaning, EDA, and baseline modeling
- Stage 2: model optimization and performance comparison
- Stage 3: minimal Dash application with visualization and prediction

## 1) Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

## 2) Get Dataset

Option A (recommended): Download manually from Kaggle and place CSV at:
- `data/avocado.csv`

Dataset URL:
- <https://www.kaggle.com/datasets/neuromusic/avocado-prices>

## 3) Run Training + Comparison

```bash
python src/train.py --data_path data/avocado.csv --target AveragePrice
```

Notes:
- Default uses a 6,000-row sample for faster iteration.
- Use full dataset with: `python src/train.py --data_path data/avocado.csv --target AveragePrice --sample_size 0`
- Include Stage 2 tuning + k-fold CV with: `python src/train.py --data_path data/avocado.csv --target AveragePrice --full_eval`

Outputs:
- `artifacts/best_model.joblib`
- `artifacts/model_report.csv`
- `artifacts/metrics_summary.json`

## 4) Run Dash App

```bash
python dash_app.py
```

App features required by the assignment:
- Uses fixed assignment dataset from `data/avocado.csv`
- Simple feature-level or whole-dataset visualization
- Manual numeric input prediction for `AveragePrice`

## Suggested Deployment

Deploy to AWS, GCP, or Heroku as required by your class.
