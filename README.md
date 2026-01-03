# Blackleg Risk Detection from UAV-Based RGB Imagery

This repository contains the code and materials associated with the Master’s thesis:

**“Blackleg Risk Detection in Industrial Potato Fields Using UAV-Based RGB Imagery and Deep Learning”**  
Author: Emil Hilligsøe Lauritsen  
Aarhus University, 2026

## Overview
This project investigates whether blackleg risk areas in industrial seed potato fields can be detected using UAV-based RGB imagery and deep learning under severe class imbalance.

The repository includes:
- The final thesis PDF
- Training and evaluation code (PyTorch)
- Experiment configurations
- Instructions for reproducing the experiments

## Repository Structure
- `thesis/` – final thesis PDF
- `code/` – model architectures, training loops, evaluation
- `configs/` – experiment configurations
- `environment/` – Python environment specifications
- `data/` – data description (raw data not publicly available)

## Data Availability
Due to confidentiality and data protection agreements, the UAV imagery and annotations used in this thesis are **not publicly available**.
The `data/README.md` describes the dataset structure and how data would be expected to be organized.

## Environment
Python 3.10  
Main dependencies:
- PyTorch
- torchvision
- Optuna
- NumPy
- scikit-learn

Install dependencies:
```bash
pip install -r environment/requirements.txt
