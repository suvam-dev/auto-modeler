import pandas as pd
import numpy as np
import os
from auto_modeler import run_quick_model

# Generate fake training data
print("Generating training data...")
N = 1000
train_df = pd.DataFrame({
    'Id': range(N),
    'Feature': np.random.randn(N),
    'Target': np.random.randn(N)
})
train_df.to_csv('dummy_train.csv', index=False)

# Train model
print("Training model...")
model = run_quick_model(
    csv_path='dummy_train.csv',
    target_col='Target',
    save_path='dummy_model.joblib',
    model_type='linear_reg'
)

# Test predict_and_save formatting
print("Testing predict_and_save formatting...")
test_df = pd.DataFrame({
    'Id': range(N, N + 100),
    'Secondary_Id': range(N*2, N*2 + 100),
    'Feature': np.random.randn(100)
})
test_df.to_csv('dummy_test.csv', index=False)

def custom_transform(x):
    return x * 100

model.predict_and_save(
    test_csv_path='dummy_test.csv',
    output_csv_path='submission.csv',
    output_target_col='SalePrice',
    keep_cols=['Id', 'Secondary_Id'],
    transform_func=custom_transform
)

# Verify output
sub_df = pd.read_csv('submission.csv')
print("\nFinal Submission Columns:", sub_df.columns.tolist())
print(sub_df.head(3))
assert list(sub_df.columns) == ['Id', 'Secondary_Id', 'SalePrice'], "Columns do not match expected Kaggle format"
print("Verification complete!")
