# Auto Modeler

A streamlined, robust wrapper around `pandas` and `scikit-learn` for rapid ML prototyping.

## Installation

### From GitHub (use in any project)

```bash
pip install git+https://github.com/suvam-dev/auto-modeler.git
```

### Local / development

```bash
git clone https://github.com/suvam-dev/auto-modeler.git
cd auto-modeler
pip install -e .
```

---

## Quick Start

### One-liner (recommended)

```python
from auto_modeler import run_quick_model

model = run_quick_model(
    csv_path='data/train.csv',
    target_col='revenue',
    save_path='models/revenue_model.joblib',
    model_type='random_forest_reg',   # default
    nan_strategy='median',            # default
)
```

### Step-by-step

```python
from auto_modeler import QuickModel

model = QuickModel(model_type='random_forest_reg', nan_strategy='median')
model.train(csv_path='data/train.csv', target_col='revenue')
model.save_model('models/revenue_model.joblib')
```

Or via the `run()` method:

```python
model = QuickModel(model_type='binary_clf')
model.run('data/churn.csv', target_col='churned', save_path='models/churn.joblib')
```

---

## API Reference

### `run_quick_model(csv_path, target_col, save_path, model_type, nan_strategy)`

| Parameter      | Type  | Default               | Description                              |
| -------------- | ----- | --------------------- | ---------------------------------------- |
| `csv_path`     | `str` | —                     | Path to the training CSV                 |
| `target_col`   | `str` | —                     | Column to predict                        |
| `save_path`    | `str` | —                     | Output path for the `.joblib` artifact   |
| `model_type`   | `str` | `'random_forest_reg'` | Algorithm (see table below)              |
| `nan_strategy` | `str` | `'median'`            | Missing-value strategy (see table below) |

### `QuickModel` methods

| Method                                  | Description                                            |
| --------------------------------------- | ------------------------------------------------------ |
| `.train(csv_path, target_col)`          | Loads CSV, builds preprocessor, fits the model         |
| `.run(csv_path, target_col, save_path)` | `train()` + `save_model()` in one call. Returns `self` |
| `.predict(new_data_csv_path)`           | Runs predictions on a new CSV. Returns a NumPy array   |
| `.save_model(filepath)`                 | Serialises the trained pipeline to disk                |
| `.load_model(filepath)`                 | Loads a previously saved pipeline from disk            |

### `model_type` options

| Value                 | Algorithm                | Use for                    |
| --------------------- | ------------------------ | -------------------------- |
| `'binary_clf'`        | Logistic Regression      | **True / False labels**    |
| `'random_forest_clf'` | Random Forest Classifier | Multi-class classification |
| `'random_forest_reg'` | Random Forest Regressor  | Continuous values          |
| `'linear_reg'`        | Linear Regression        | Continuous values          |
| `'logistic_reg'`      | Logistic Regression      | Multi-class classification |

### `nan_strategy` options

| Value             | Behaviour                              |
| ----------------- | -------------------------------------- |
| `'median'`        | Impute numeric NaNs with column median |
| `'mean'`          | Impute numeric NaNs with column mean   |
| `'most_frequent'` | Impute with the most common value      |
| `'constant'`      | Fill with a constant value             |
| `'drop'`          | Drop rows containing any NaN           |
