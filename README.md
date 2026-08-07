# Migraine Severity Prediction Thesis

This project investigates whether physiological and environmental
factors can predict migraine severity using machine learning.

## Target

Migraine severity is classified using a VAS-derived target:

- Mild: VAS 1–3
- Moderate: VAS 4–6
- Severe: VAS 7–10

## Study population

The modeling dataset includes diary observations that:

1. Meet ICHD-3 criteria for migraine.
2. Contain a valid VAS-derived severity label.

## Models

- Logistic Regression
- Random Forest
- XGBoost

## Feature groups

- Physiological
- Environmental
- Lifestyle
- Temporal
- Historical weather (from API)

## Project structure

- `Data/Raw`: original source data
- `Data/Intermediate`: exported or partially transformed files
- `Data/Processed`: cleaned datasets
- `Notebooks`: sequential analysis notebooks
- `src`: reusable Python code and configuration
- `Figures and Tables`: thesis-ready figures and tables
- `Results`: analytical outputs and model results
- `Models`: saved trained models

## Notebook order

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Development
6. Model Evaluation