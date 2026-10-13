# ml-01 — ML Foundations

Part of my AI Engineering transition — 20-week structured roadmap from Technical Director to AI Powered Engineer/Architect.

## My AI engineering roadmap
Building in public — Technical Director transitioning to AI Engineer.

| Repo | Covers | Status |
|------|--------|--------|
| [ml-01-practice](https://github.com/srastogi2/ml-01-python-ml-foundations) | pandas, NumPy, matplotlib, seaborn, scikit-learn, TensorFlow | ✅ Done |
| llm-01-qa-bot | LLM APIs, prompt engineering | 🔜 Week 5 |
| llm-02-rag-pipeline | LangChain, vector DBs, evaluation — data integration domain | 🔜 Week 8 |
| llm-03-agent | LangGraph, tool use, MCP | 🔜 Week 10 |

## Courses completed
- Python for Data Science and Machine Learning Bootcamp — Jose Portilla (Udemy)
- Supervised Machine Learning: Regression and Classification — Andrew Ng, DeepLearning.AI & Stanford (Coursera)

## Projects in this repo

1. **pandas & NumPy basics** — data manipulation fundamentals
2. **Data visualisation** — matplotlib & seaborn
3. **Advertising classifier** — Logistic Regression, 0.926 accuracy
4. **Titanic classifier** — first end-to-end ML workflow, 0.76 on Kaggle leaderboard
5. **LendingClub loan repayment** — first neural network, TensorFlow/Keras

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

## 3. Advertising classifier — Logistic Regression

Built a classifier to predict whether a user clicks on an online advertisement, using the bootcamp's advertising dataset — the course's guided Logistic Regression project.

### Workflow
```
Load dataset → EDA (age, income, daily internet usage vs click behaviour) → 
Feature selection → Train/test split → Train Logistic Regression → 
Evaluate (accuracy, classification report)
```

### Results
Achieved **0.926 accuracy** on the held-out test set, with a strong classification report across both classes (clicked / did not click).

### Key learnings
- This dataset is far more separable than Titanic — user behaviour features (time on site, internet usage, age) correlate strongly with the click outcome, which explains the higher accuracy
- Confirmed the full supervised learning workflow a second time on a cleaner dataset, reinforcing the train/test split → fit → predict → evaluate pattern
- Comparing this project's 0.926 to Titanic's 0.76 was a useful lesson: accuracy is dataset-dependent, not just model-dependent — a "good" score means different things on different problems

### What surprised me
How much a clean, well-separated dataset improves results versus real-world messy data like Titanic — a preview of why data quality work matters more than model choice in many real projects.

---

## 4. Titanic classifier — first end-to-end ML workflow

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

## 5. LendingClub loan repayment — first neural network

Built a binary classifier in **TensorFlow/Keras** to predict whether a LendingClub borrower repays their loan, following the bootcamp's guided deep learning project.

### Workflow
```
Load LendingClub data → EDA (loan status balance, feature correlations) → 
Handle missing values & drop leaky columns → Encode categoricals (dummies) → 
Train/test split → Scale features (MinMaxScaler) → 
Build Sequential model (Dense + Dropout) → Train with validation data → 
Evaluate (loss curves, classification report, confusion matrix)
```

### Key learnings
- Keras follows a clear build → compile → fit → evaluate pattern, close to scikit-learn's fit/predict but with explicit control over layers, loss and optimiser
- Feature scaling is not optional for neural networks — unscaled inputs slow or destabilise training
- Plotting training vs validation loss per epoch is the quickest way to spot overfitting; Dropout layers and early stopping are the standard fixes
- The dataset is imbalanced (most loans are repaid), so the classification report matters more than headline accuracy — the hard part is catching the minority class of defaults

### What surprised me
How much of the work is data preparation rather than the network itself — cleaning, encoding and removing columns that leak the outcome took far longer than defining the model.

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
03-advertising-classifier/
04-titanic-classifier/
05-lendingclub-tensorflow/
```

## Stack
- Python 3.11
- pandas, NumPy, matplotlib, seaborn, scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## License
MIT
