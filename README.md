As part of Kaggle Competition for my MSBA program at the University of Virginia, I employed complex NLP techniques to determine if 
an answer to a medical question on an online forum was from an expert or from a non-expert.

Here is what I did:

# RateMyAnswer: Expert vs. Non-Expert Health Answer Classifier

A machine learning pipeline that classifies whether a health forum answer was written by an expert or a non-expert, optimized for log-loss. This notebook won a private class Kaggle competition with a final log-loss of **0.03307**.

---

## Overview

The task was to predict the probability that a given health forum answer came from an expert, scored using log-loss. The winning approach used a Logistic Regression model with TF-IDF features and hand-engineered linguistic signals — no GPU, no transformers, no ensembling.

The key insight was that Logistic Regression directly minimizes log-loss as its objective function, making it natively better calibrated than LSTMs or tree-based models for this metric.

---

## Approach

### Preprocessing
- Byte string normalization (forum data was stored with `b'...'` prefixes)
- Light cleaning: HTML tag removal, URL normalization, whitespace normalization
- Selective stop word removal — preserving high-signal words like negations (`not`, `never`) and hedges (`may`, `might`, `consult`)

### Feature Engineering
Hand-crafted features designed to capture the linguistic fingerprint of expert health writing:

- **Readability metrics** — Flesch-Kincaid grade, Gunning Fog index
- **Medical vocabulary ratio** — proportion of recognized clinical terms
- **Hedge word ratio** — epistemic hedging language common in expert writing
- **Short sentence ratio** — non-experts tend to write clipped, fragmented responses
- **Unique word ratio** — lexical diversity as a proxy for writing sophistication
- **Structural signals** — punctuation patterns, sentence length variance

### Vectorization
- Word TF-IDF with n-grams (1–3), capturing expert phrasing patterns
- Character TF-IDF with n-grams (2–6), capturing medical morphology (e.g. `-itis`, `cardio-`, `prescri-`)
- Combined with scaled hand-engineered features via sparse matrix stacking

### Model
- Logistic Regression with L2 regularization (`C=275`, `max_iter=4500`, `solver='saga'`)
- `class_weight='balanced'` to account for 72/28 class imbalance
- C value selected via 5-fold stratified cross-validation optimizing log-loss directly

---

## Results

| Metric | Score |
|---|---|
| Competition log-loss | **0.03307** |
| Final placement | **1st place** |

---

## Repository Structure

```
├── notebook.ipynb       # Full pipeline — preprocessing through submission
└── README.md
```

The datasets are not included in this repository. If you are interested in the data, feel free to reach out directly.

---

## Requirements

```
pandas
numpy
scikit-learn
scipy
textstat
```

---

## Contact

Questions about the dataset or methodology — feel free to message me.
