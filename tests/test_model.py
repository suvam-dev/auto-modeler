import unittest
import pandas as pd
import numpy as np
import os
from src import QuickModel

class TestQuickModel(unittest.TestCase):
    
    def setUp(self):
        """Create a temporary CSV for testing before each test runs."""
        self.test_csv = 'test_data.csv'
        data = {
            'feature1': [1, 2, np.nan, 4],
            'feature2': ['A', 'B', 'A', 'B'],
            'target': [10, 20, 15, 30]
        }
        pd.DataFrame(data).to_csv(self.test_csv, index=False)
        
    def tearDown(self):
        """Clean up the temporary CSV after each test runs."""
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
            
    def test_initialization_invalid_model(self):
        """Ensure it raises an error if an unsupported model is passed."""
        with self.assertRaises(ValueError):
            model = QuickModel(model_type='unsupported_model')
            model._get_estimator()

    def test_training_pipeline_with_median_imputation(self):
        """Test if the model trains successfully using median imputation."""
        model = QuickModel(model_type='linear_reg', nan_strategy='median')
        model.train(self.test_csv, target_col='target')
        self.assertIsNotNone(model.pipeline)

if __name__ == '__main__':
    unittest.main()