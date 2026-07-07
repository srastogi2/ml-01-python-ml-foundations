# ml-01 — pandas & NumPy Fundamentals

Part of my AI Engineering transition — Roadmap from Technical Director to AI Engineer.

## What I worked on
- **NumPy** — arrays, indexing, slicing, broadcasting, vectorised operations
- **pandas** — DataFrames, filtering, groupby, summary statistics
- **Mini exercise** — loaded a CSV, cleaned it (nulls, type fixes), computed descriptive stats

## Key learnings
- NumPy vectorised operations are significantly faster than Python loops — critical to internalise before writing ML code
- Pandas `groupby` + `agg` maps closely to SQL GROUP BY with multiple aggregations — useful mental model coming from a SWE background
- Matplotlib foundational plotting library; fine-grained control over every chart element but verbose for quick exploration; 
- Seaborn builds on Matplotlib with cleaner syntax for statistical plots;
  heatmaps and correlation matrices with a single line vs many lines in raw Matplotlib
- Plotting distributions with histograms early in data exploration reveals skew and outliers  before any modelling — something I'd previously left to data science teams
- Correlation heatmaps give an instant visual of feature relationships; useful signal for feature selection before feeding data into a model

## What surprised me
The indexing difference between NumPy (position-based) and pandas (label-based `.loc` / `.iloc`) tripped me up early. Worth remembering as notebooks grow in complexity.
Seaborn's `pairplot()` generates a full matrix of scatter plots across all numeric columns in one line — the kind of EDA(Exploratory Data Analysis) that would have taken significant effort without it.


## Design decisions
- Chose Jupyter Notebook over a plain `.py` script — easier to annotate learnings inline alongside code, which matters more at this stage than production structure
- Kept the CSV generic (sample data) to focus on library mechanics rather than domain-specific cleaning rules

## How to run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Stack (refer requirements.txt)
- Python 3.13.9
- pandas 2.x · NumPy 1.x · Jupyter
- Matplotlib 
- Seaborn

## Roadmap
## Part of my AI Engineering roadmap
| Repo | Status |
|------|--------|
| ml-01-python-ml-foundations | ✅ Current |
