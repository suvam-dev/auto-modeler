import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

class QuickModel:
    """
    A unified wrapper for automating pandas preprocessing and scikit-learn model training.
    """
    
    def __init__(self, model_type: str = 'random_forest_reg', nan_strategy: str = 'median'):
        """
        Initializes the model configuration.
        
        Args:
            model_type (str): The algorithm to use. 
                              Options: 'linear_reg', 'logistic_reg', 'random_forest_reg', 'random_forest_clf'.
            nan_strategy (str): How to handle missing values. 
                                Options: 'drop', 'mean', 'median', 'most_frequent', 'constant'.
        """
        self.model_type = model_type
        self.nan_strategy = nan_strategy
        self.pipeline = None
        self.target_col = None

    def _get_estimator(self):
        """Maps the string model_type to the actual scikit-learn model object."""
        models = {
            'linear_reg': LinearRegression(),
            'logistic_reg': LogisticRegression(max_iter=1000),
            'binary_clf': LogisticRegression(max_iter=1000),   # explicit alias for True/False labels
            'random_forest_reg': RandomForestRegressor(random_state=42),
            'random_forest_clf': RandomForestClassifier(random_state=42),
        }
        
        if self.model_type not in models:
            raise ValueError(f"Model '{self.model_type}' not supported. Choose from: {list(models.keys())}")
        
        return models[self.model_type]

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Dynamically detects column types and builds a preprocessing pipeline.
        Numeric columns get imputed and scaled.
        Categorical columns get imputed and one-hot encoded.
        Boolean columns are cast to int (1/0) and treated as numeric.
        """
        # Cast bool columns to int so sklearn doesn't trip on True/False dtype
        bool_cols = X.select_dtypes(include='bool').columns
        if len(bool_cols):
            X = X.copy()
            X[bool_cols] = X[bool_cols].astype(int)

        numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'uint8']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        # Determine imputation strategy based on user input
        impute_strategy = self.nan_strategy if self.nan_strategy != 'drop' else 'median'

        # Numeric Pipeline: Impute -> Standardize
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('scaler', StandardScaler())
        ])

        # Categorical Pipeline: Impute -> Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine into a single transformer
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def train(self, csv_path: str, target_col: str):
        """
        Loads CSV data, applies NaN strategies, builds the pipeline, and trains the model.
        
        Args:
            csv_path (str): The file path to the training CSV data.
            target_col (str): The name of the column to predict.
        """
        self.target_col = target_col
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # 1. Handle 'drop' strategy at the pandas DataFrame level
        if self.nan_strategy == 'drop':
            initial_rows = df.shape[0]
            df = df.dropna()
            print(f"Dropped {initial_rows - df.shape[0]} rows containing NaN values.")

        # 2. Split into features (X) and target (y)
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in CSV.")
            
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Cast bool target (True/False) to int so classifiers receive 0/1
        if y.dtype == bool:
            y = y.astype(int)

        # 3. Construct the full pipeline
        preprocessor = self._build_preprocessor(X)
        estimator = self._get_estimator()

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', estimator)
        ])

        # 4. Train the pipeline
        print(f"Training {self.model_type}...")
        self.pipeline.fit(X, y)
        print("Training complete! Model is ready for predictions.")

    def predict(self, new_data_csv_path: str) -> np.ndarray:
        """Loads new CSV data and generates predictions."""
        if self.pipeline is None:
            raise Exception("Model not trained. Call train() first.")
            
        new_df = pd.read_csv(new_data_csv_path)
        return self.pipeline.predict(new_df)

    def save_model(self, filepath: str):
        """Serializes the trained pipeline to disk."""
        if self.pipeline is None:
            raise Exception("No model to save.")
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved securely to {filepath}")
        
    def load_model(self, filepath: str):
        """Loads a previously saved pipeline from disk."""
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def run(self, csv_path: str, target_col: str, save_path: str) -> 'QuickModel':
        """
        Super-function: trains the model on *csv_path* and saves it to *save_path*
        in one call.  Returns `self` so the instance can be used immediately.

        Args:
            csv_path  (str): Path to the training CSV file.
            target_col (str): Name of the column to predict.
            save_path  (str): Destination path for the saved model artifact
                              (e.g. 'models/revenue_model.joblib').

        Returns:
            QuickModel: The trained instance (for optional method chaining).

        Example::

            model = QuickModel(model_type='random_forest_reg', nan_strategy='median')
            model.run('data/my_data.csv', target_col='revenue', save_path='models/revenue_model.joblib')

            # — or in one line —
            model = run_quick_model('data/my_data.csv', target_col='revenue',
                                    save_path='models/revenue_model.joblib')
        """
        self.train(csv_path=csv_path, target_col=target_col)
        self.save_model(filepath=save_path)
        return self


# ---------------------------------------------------------------------------
# Module-level convenience wrapper
# ---------------------------------------------------------------------------

def run_quick_model(
    csv_path: str,
    target_col: str,
    save_path: str,
    model_type: str = 'random_forest_reg',
    nan_strategy: str = 'median',
) -> QuickModel:
    """
    One-liner super-function: initialises, trains, and saves a model.

    Args:
        csv_path    (str): Path to the training CSV file.
        target_col  (str): Name of the column to predict.
        save_path   (str): Destination path for the saved model artifact.
        model_type  (str): Algorithm to use (default: 'random_forest_reg').
        nan_strategy (str): Missing-value strategy (default: 'median').

    Returns:
        QuickModel: The fully trained instance.

    Example::

        from src import run_quick_model

        model = run_quick_model(
            csv_path='data/my_data.csv',
            target_col='revenue',
            save_path='models/revenue_model.joblib',
        )
    """
    model = QuickModel(model_type=model_type, nan_strategy=nan_strategy)
    return model.run(csv_path=csv_path, target_col=target_col, save_path=save_path)