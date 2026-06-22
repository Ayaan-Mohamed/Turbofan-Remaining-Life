# Turbofan Remaining Life

## Problem
This project builds a machine learning classifier to predict imminent engine failure using the NASA C-MAPSS turbofan dataset. The model analyzes multi-sensor telemetry from operating engines to identify those approaching the end of their useful life, so maintenance can be scheduled proactively rather than reactively.

## Dataset
NASA C-MAPSS (FD001 subset), covering the operational lifetime of 100 simulated turbofan engines.
Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Note: the `data/` folder is gitignored, so it is not included in this repo. Download the dataset from the link above and place the `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` files in a `data/` folder at the project root.

## Setup
Requires pandas, numpy, scikit-learn, and matplotlib (see `requirements.txt`).
The notebooks must be run in order (`01`, `02`, `03`), as each one generates data that the next depends on.

## Approach
1. Explored the data and removed missing values and zero-variation sensors.
2. Labeled each cycle as failing or healthy based on remaining useful life.
3. Split the data with GroupKFold to prevent leakage across engines.
4. Trained and tuned five models, comparing them on F2 score and runtime.

## Results
All models were tuned with 5-fold GroupKFold, optimizing F2.

| Model                                  | F2 score | Spread (std) | Tuning time |
|----------------------------------------|----------|--------------|-------------|
| Logistic Regression                    | 0.8923   | 0.0103       | 45.6s       |
| K-Nearest Neighbors                    | 0.8449   | 0.0332       | 10.9s       |
| Support Vector Classifier (radial)     | 0.8978   | 0.0129       | 15m 50.6s   |
| Random Forest Classifier               | 0.8730   | 0.0238       | 5m 26.3s    |
| Histogram Gradient Boosting Classifier | 0.8973   | 0.0185       | 1m 24.6s    |

There was a three-way tie between Logistic Regression, the Support Vector Classifier (SVC), and the Histogram Gradient Boosting Classifier, which all scored very similarly (in order: SVC, Histogram Gradient Boosting, Logistic Regression). The differences between their scores were smaller than the fold-to-fold spread, so they were effectively tied. However, the SVC took significantly longer to tune than the other two, making Logistic Regression and Histogram Gradient Boosting Classifier the winning options.

## Key Decisions
**GroupKFold over a standard train/test split:** because each engine is measured many times, an engine's measurements appearing in both training and test data would be a form of leakage. GroupKFold avoids this by keeping all of an engine's measurements entirely in either training or validation.

**F2 score over F1:** identifying every failing engine matters more than avoiding the occasional false alarm, since a missed failure is far more dangerous than a false positive. Prioritizing recall over precision is captured by the F2 score (beta = 2), which weighs both recall and precision while favoring recall.

**Excluding SVC from the winners:** its small scoring advantage was within the fold-to-fold spread and not statistically meaningful, so it was not worth its dramatically longer runtime.

## Project Structure
```
├── data/                  (gitignored, add dataset here)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_labeling_and_baseline.ipynb
│   └── 03_models_and_tuning.ipynb
├── src/
│   └── model_selection.py
├── requirements.txt
└── README.md
```