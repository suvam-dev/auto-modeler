import pandas as pd
import numpy as np
from src import QuickModel

def main():
    # 1. Generate a mock CSV with missing data to test the pipeline
    print("Generating sample data (data/sample_data.csv)...")
    data = {
        'age': [25, 30, np.nan, 45, 35, 50, 28, 40],
        'city': ['NYC', 'LA', 'NYC', np.nan, 'SF', 'SF', 'LA', 'NYC'],
        'experience_years': [2, 5, 4, 15, 8, 20, 3, 10],
        'salary': [60000, 80000, 75000, 120000, 95000, 140000, 65000, 110000]
    }
    df = pd.DataFrame(data)
    df.to_csv('data/sample_data.csv', index=False)

    # 2. Initialize the architecture
    # We want a Random Forest Regressor and we want to impute NaNs with the median value
    print("\n--- Initializing Model ---")
    modeler = QuickModel(model_type='random_forest_reg', nan_strategy='median')

    # 3. Train on the CSV
    print("\n--- Starting Training ---")
    modeler.train(csv_path='data/sample_data.csv', target_col='salary')

    # 4. Save the compiled model
    print("\n--- Saving Artifacts ---")
    modeler.save_model('models/salary_predictor.joblib')

if __name__ == "__main__":
    main()