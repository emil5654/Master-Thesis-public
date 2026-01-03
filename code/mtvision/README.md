# mtvision

`mtvision` is the internal research framework used in this Master’s thesis to run the full
machine learning pipeline for blackleg risk detection from UAV-based imagery.

The framework covers the complete workflow:

**dataset loading → model creation → training → evaluation**

This codebase is structured as a small research framework rather than a single script.
If you are new to the project, start with **Where to start** below.

---

## Repository context

- The dataset used in this thesis is **not publicly available** 
- The code is provided for transparency and reproducibility **given data access**.
- Preprocessing generates tile-based datasets organised as:


```text
OUT_ROOT/
├── dataset_0000/
│   ├── High Risk/ (*.png or *.npy)
│   └── No Risk/   (*.png or *.npy)
└── ...


---

## Folder overview

Typical structure inside `mtvision/`:

- **data/**
  - Dataset classes and utilities
  - Tile loading, transforms, class labels, and split logic

- **models/**
  - Model definitions and factory logic
  - Custom CNNs, ResNet-18, and Vision Transformer wrappers

- **training/**
  - Training loops and optimisation logic
  - Loss functions, metrics, samplers, and logging utilities

- **hyperparameter_tuning/**
  - Scripts used for Optuna-based hyperparameter optimisation

- **testset_evaluation/**
  - Scripts used for final evaluation on the held-out test set

---

## Where to start

If you want to understand the pipeline end-to-end, the recommended reading order is:

1. **Preprocessing**
   - Inspect the preprocessing logic in `data/preprocess/`
   - Understand how orthomosaics and annotations are converted into tile-based datasets

2. **Dataset implementation**
   - Inspect the main dataset class in `mtvision/data/`
   - Focus on expected folder structure, label encoding, and spatial split logic

3. **Model factory**
   - Review how model names map to implementations in `models/`
   - This is where architectural choices are centralised

4. **Training loop**
   - Follow the main training functions in `training/`
   - Observe how loss functions, optimisers, and metrics are applied

5. **Evaluation and tuning**
   - Inspect `hyperparameter_tuning/` to see how model configurations are optimised
   - Inspect `testset_evaluation/` to see how final performance on the held-out test set is computed

---

## Project conventions

### Labels
Binary classification is used throughout the project:

- **Risk** (minority class)
- **No Risk** (majority class)

### Evaluation focus
The thesis prioritises minority-class performance. Reported metrics therefore emphasise:

- F1_risk
- Recall_risk
- Precision_risk
- Overall accuracy (for completeness)

### Class imbalance handling
The primary strategy evaluated in the thesis is:

- Sampling-based class balancing using a weighted sampler
- Cross-entropy loss

Alternative strategies may appear in archived or exploratory scripts.

### Data splitting
To reduce spatial leakage, dataset splits are performed using spatial blocking or
group-based splitting where applicable.

---

## Running the code (high level)

Because the dataset is not public, running the code requires access to a compatible
local dataset layout.

Typical workflow:

1. Preprocess UAV imagery to generate tile-based datasets  
   (see code in `data/preprocess/`)
2. Insert configurations into `testset_evaluation/` or `hyperparameter_tuning/` scripts
3. Run the corresponding experiment or evaluation script

