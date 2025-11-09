# Poker Hand ML Project

This repository evaluates multiple classifiers (KNN, Random Forest, Gradient Boosting, XGBoost) on the UCI Poker Hand dataset and on a **balanced synthetic dataset** to demonstrate the impact of **class imbalance** on evaluation metrics (Accuracy vs. Macro/Weighted F1).

## Structure
- `data/`
  - `poker-hand-training-true.data` (place here; original UCI dataset)
  - `poker_balanced_10k.csv` (created by `01_generate_balanced_dataset.ipynb` or `src/generate_balanced_dataset.py`)
- `notebooks/`
  - `01_generate_balanced_dataset.ipynb`
  - `02_train_models_imbalanced.ipynb`
  - `03_train_models_balanced.ipynb`
  - `04_compare_models.ipynb`
- `src/`
  - `generate_balanced_dataset.py`
  - `utils_metrics.py`
  - `run_experiment.py`
- `results/`
  - `confusion_matrices/`
  - `f1_barplots/`
  - `model_comparison.csv`
- `docs/`
  - `research_summary.md`
  - `experiment_log.md`

## Quick Start
1. Place the original UCI file into `data/poker-hand-training-true.data`.
2. Open `notebooks/01_generate_balanced_dataset.ipynb` and run all cells to create `data/poker_balanced_10k.csv`.
3. Run `02_train_models_imbalanced.ipynb` (uses UCI data).
4. Run `03_train_models_balanced.ipynb` (uses balanced data).
5. Run `04_compare_models.ipynb` to see side-by-side comparisons.

---
Generated: 2025-11-09T20:56:51.060707
