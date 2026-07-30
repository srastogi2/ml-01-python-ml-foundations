# ml-01 — Python ML Foundations

Part of my AI Engineering transition — 20-week structured roadmap from Technical Director to AI Engineer.

## My AI engineering roadmap
Building in public — Technical Director transitioning to AI Engineer.

| Repo | Covers | Status |
|------|--------|--------|
| [ml-01-python-ml-foundations](https://github.com/srastogi2/ml-01-python-ml-foundations) | pandas, NumPy, matplotlib, seaborn, scikit-learn | ✅ Done 

## Projects in this repo

1. **pandas & NumPy basics** — data manipulation fundamentals
2. **Data visualisation** — matplotlib & seaborn
3. **Titanic classifier** — first end-to-end ML workflow, 0.76 on Kaggle leaderboard

---

## 1. pandas & NumPy basics

- **NumPy** — arrays, indexing, slicing, broadcasting, vectorised operations
- **pandas** — DataFrames, filtering, groupby, summary statistics
- **Mini exercise** — loaded a CSV, cleaned it (nulls, type fixes), computed descriptive stats

### Key learnings
- NumPy vectorised operations are significantly faster than Python loops — critical to internalise before writing ML code
- pandas `groupby` + `agg` maps closely to SQL GROUP BY with multiple aggregations — useful mental model coming from a SWE background
- Both libraries were largely new territory; more depth here than expected

### What surprised me
The indexing difference between NumPy (position-based) and pandas (label-based `.loc` / `.iloc`) tripped me up early. Worth remembering as notebooks grow in complexity.

---

## 2. Data visualisation — matplotlib & seaborn

- **Matplotlib** — foundational plotting library; fine-grained control over every chart element but verbose for quick exploration; best used when precise customisation is needed
- **Seaborn** — builds on Matplotlib with cleaner syntax for statistical plots; heatmaps and correlation matrices in a single line vs many lines in raw Matplotlib
- Plotted distributions, bar charts, histograms, scatter plots, heatmaps and correlation matrices

### Key learnings
- Plotting distributions with histograms early in data exploration reveals skew and outliers before any modelling — something I'd previously left to data science teams
- Correlation heatmaps give an instant visual of feature relationships — useful signal for feature selection before feeding data into a model
- This process of visually/statistically examining data before modelling is called **EDA (Exploratory Data Analysis)** — a standard first step in any ML workflow

### What surprised me
Seaborn's `pairplot()` generates a full matrix of scatter plots across all numeric columns in one line — the kind of EDA that would have taken significant manual effort without it.

---

## 3. Titanic classifier — first end-to-end ML workflow

Built a survival classifier on the Kaggle Titanic dataset using **Logistic Regression**, covering the complete ML workflow from raw data to a scored competition submission.

### Workflow
```
Load train.csv → EDA → Clean & impute missing values → 
Feature selection → Train/validation split → Train Logistic Regression → 
Validate (accuracy, classification report) → Retrain on full train.csv → 
Predict on test.csv → Submit to Kaggle
```

### Results
Achieved **0.76 accuracy** on the Kaggle public leaderboard (rank ~9,489), covering the full ML workflow: EDA, missing-value imputation, feature selection, train/validation split, model training, and competition submission.

### Key learnings
- Kaggle's `test.csv` has no target column by design — validation must be done on a held-out split of `train.csv`, not on `test.csv`
- `dropna()` is safe on training data but breaks submissions on test data — every row must get a prediction, so missing values need to be **imputed**, not dropped
- Accuracy alone can be misleading on imbalanced data; the classification report and confusion matrix show precision/recall balance, not just a single number
- The metric that matters depends on the problem — recall matters more in some domains (e.g. disease screening), precision in others (e.g. spam detection)

### Design decisions
- Chose Logistic Regression as a first model — simple, interpretable, and a natural baseline before trying ensemble methods
- Filled missing `Age` and `Fare` values with median rather than dropping rows, to preserve all 418 required predictions for submission

### What surprised me
A meaningful share of very high Titanic leaderboard scores come from lookups against the real historical passenger record, since the dataset is public history — so 0.76–0.80 is a realistic, honest range for a properly trained model.

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Each project lives in its own subfolder:
```
01-pandas-numpy/
02-visualization/
03-titanic-classifier/
```

## Stack
- Python 3.11
- pandas, NumPy, matplotlib, seaborn, scikit-learn
- Jupyter Notebook

## License
MIT
