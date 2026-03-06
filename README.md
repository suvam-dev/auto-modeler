# Auto Modeler

A streamlined, robust wrapper around `pandas` and `scikit-learn` for rapid machine learning prototyping.

## Features

- Pass a CSV, a target column, and a model type — the pipeline automatically handles data typing, imputation, and scaling.
- Prevents data leakage via strict `scikit-learn` Pipelines.
- Supports regression and classification out of the box.
- Export ready-to-use `.joblib` model artifacts.

---

## Quick Start

### One-liner (recommended)

```python
from src import run_quick_model

model = run_quick_model(
    csv_path='data/my_data.csv',
    target_col='revenue',
    save_path='models/revenue_model.joblib',
    model_type='random_forest_reg',   # default
    nan_strategy='median',            # default
)
```

`run_quick_model` initialises, trains, and saves the model in a single call. It returns a trained `QuickModel` instance that can be used for predictions immediately.

---

### Step-by-step

```python
from src import QuickModel

# 1. Initialise
model = QuickModel(model_type='random_forest_reg', nan_strategy='median')

# 2. Train
model.train(csv_path='data/my_data.csv', target_col='revenue')

# 3. Save
model.save_model('models/revenue_model.joblib')
```

Or use the `run()` method on an existing instance:

```python
model = QuickModel(model_type='linear_reg', nan_strategy='mean')
model.run('data/my_data.csv', target_col='revenue', save_path='models/revenue_model.joblib')
```

---

## API Reference

### `run_quick_model(csv_path, target_col, save_path, model_type, nan_strategy)`

| Parameter      | Type  | Default               | Description                            |
| -------------- | ----- | --------------------- | -------------------------------------- |
| `csv_path`     | `str` | —                     | Path to the training CSV               |
| `target_col`   | `str` | —                     | Column to predict                      |
| `save_path`    | `str` | —                     | Output path for the `.joblib` artifact |
| `model_type`   | `str` | `'random_forest_reg'` | Algorithm to use (see below)           |
| `nan_strategy` | `str` | `'median'`            | Missing-value strategy (see below)     |

### `QuickModel(model_type, nan_strategy)`

| Method                                  | Description                                            |
| --------------------------------------- | ------------------------------------------------------ |
| `.train(csv_path, target_col)`          | Loads CSV, builds preprocessor, and fits the model     |
| `.run(csv_path, target_col, save_path)` | `train()` + `save_model()` in one call. Returns `self` |
| `.predict(new_data_csv_path)`           | Runs predictions on a new CSV. Returns a NumPy array   |
| `.save_model(filepath)`                 | Serialises the trained pipeline to disk                |
| `.load_model(filepath)`                 | Loads a previously saved pipeline from disk            |

### Supported `model_type` values

| Value                 | Algorithm                | Use for                 |
| --------------------- | ------------------------ | ----------------------- |
| `'binary_clf'`        | Logistic Regression      | **True / False labels** |
| `'random_forest_reg'` | Random Forest Regressor  | Continuous values       |
| `'random_forest_clf'` | Random Forest Classifier | Multi-class labels      |
| `'linear_reg'`        | Linear Regression        | Continuous values       |
| `'logistic_reg'`      | Logistic Regression      | Multi-class labels      |

### Supported `nan_strategy` values

| Value             | Behaviour                              |
| ----------------- | -------------------------------------- |
| `'median'`        | Impute numeric NaNs with column median |
| `'mean'`          | Impute numeric NaNs with column mean   |
| `'most_frequent'` | Impute with the most common value      |
| `'constant'`      | Fill with a constant value             |
| `'drop'`          | Drop rows containing any NaN           |

---

## Running the demo

```bash
python main.py
```

This generates `data/sample_data.csv`, trains a Random Forest Regressor on the `salary` column, and saves the artifact to `models/salary_predictor.joblib`.
