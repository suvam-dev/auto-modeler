import pandas as pd
import numpy as np
from auto_modeler import run_quick_model

print("Generating data...")
N = 10000
df = pd.DataFrame({
    'Patient_ID': [f'P_{i}' for i in range(N)],
    'Age': np.random.randint(20, 80, N),
    'Cholesterol': np.random.randint(150, 300, N),
    'Heart Disease': np.random.choice(['Yes', 'No'], N)
})
df.to_csv('dummy_train.csv', index=False)

print("Running model...")
model = run_quick_model(
    csv_path='dummy_train.csv',
    target_col='Heart Disease',
    save_path='dummy_model.joblib',
    model_type='random_forest_reg'
)
print("Done!")
