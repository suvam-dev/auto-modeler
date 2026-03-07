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
    ===========================================================================
    🤖 QuickModel: The Automated Machine Learning Engine
    ===========================================================================
    
    A unified, high-level wrapper designed to automate pandas preprocessing 
    and scikit-learn model training into a seamless, robust pipeline.
    
    Features:
        • Automatic data-type detection & missing value imputation
        • Data leakage prevention via strict scikit-learn Pipelines
        • Direct `.joblib` model artifact serialization
    """
    def __init__(self, model_type: str = 'random_forest_reg', nan_strategy: str = 'median'):
        """
        ⚡ Initializes the ML engine configuration.
        
        Args:
            model_type   (str): The underlying scikit-learn estimator algorithm. 
                                [Supported: 'linear_reg', 'logistic_reg', 
                                 'binary_clf', 'random_forest_reg', 'random_forest_clf']
            nan_strategy (str): The strategy for handling missing values (NaNs). 
                                [Supported: 'drop', 'mean', 'median', 
                                 'most_frequent', 'constant']
        """
        self.model_type = model_type
        self.nan_strategy = nan_strategy
        self.pipeline = None
        self.target_col = None




    def _get_estimator(self):
        """
        🧠 Instantiates the core Machine Learning algorithm.
        
        This private method acts as a factory, mapping the user-provided 
        `model_type` string to its corresponding scikit-learn class.
        
        Returns:
            estimator: An un-fitted scikit-learn model object.
            
        Raises:
            ValueError: If the `model_type` is not in the supported dictionary.
        """
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
        🛠️ Dynamically constructs a preprocessing graph based on column dtypes.
        
        Logic mapping:
          • Numeric data  -> Imputation + StandardScaler
          • Categorical   -> Imputation + OneHotEncoder
          • Boolean       -> Cast to Integer (0/1) + treated as numeric
        
        Returns:
            ColumnTransformer: The compiled preprocessor node ready for fitting.
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
            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=100, sparse_output=False))
        ])

        # Combine into a single transformer
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def train(self, csv_path: str, target_col: str, max_samples: int = None):
        """
        🚀 The main execution engine for training the model.
        
        This method handles:
            1. Loading data directly from disk via pandas
            2. Dropping NaN rows natively (if requested via `nan_strategy`)
            3. Type-validating the target column against the model type
            4. Constructing the sklearn Pipeline graph
            5. Fitting the Pipeline to the historical data
        
        Args:
            csv_path    (str): The absolute or relative filepath to the training CSV.
            target_col  (str): The exact name of the column you are trying to predict.
            max_samples (int): Max number of rows to train on. Selects randomly if exceeded.
        """
        self.target_col = target_col
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # 0. Chop data if max_samples is provided
        if max_samples is not None and len(df) > max_samples:
            print(f"Dataset too large ({len(df)} rows). Randomly sampling down to {max_samples} rows...")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

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

        # 2b. Map binary string targets to 1/0
        if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
            clean_y = y.dropna().astype(str).str.lower()
            unique_vals = list(set(clean_y.unique()))
            
            mapping = None
            if len(unique_vals) == 2:
                # If it's a known boolean pair, map explicitly for predictability
                if set(unique_vals) == {'true', 'false'}:
                    mapping = {'true': 1, 'false': 0}
                elif set(unique_vals) == {'yes', 'no'}:
                    mapping = {'yes': 1, 'no': 0}
                elif set(unique_vals) == {'y', 'n'}:
                    mapping = {'y': 1, 'n': 0}
                elif set(unique_vals) == {'t', 'f'}:
                    mapping = {'t': 1, 'f': 0}
                else:
                    # Otherwise, arbitrarily map the two unique values to 0 and 1
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    print(f"Auto-mapped binary target '{self.target_col}': {unique_vals[0]} -> 0, {unique_vals[1]} -> 1")
                
            if mapping:
                y = clean_y.map(mapping).reindex(y.index)

        # Cast bool target (True/False) to int so classifiers receive 0/1
        if y.dtype == bool:
            y = y.astype(int)

        # 2c. Validate target type against model type
        is_regressor = self.model_type in ['linear_reg', 'random_forest_reg']
        if is_regressor and (y.dtype == 'object' or pd.api.types.is_string_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype)):
            raise ValueError(
                f"Target column '{self.target_col}' contains text/categories (e.g., '{y.iloc[0]}'), "
                f"but you are using a Regressor ('{self.model_type}'). "
                f"Please change model_type to a Classifier (e.g., 'random_forest_clf' or 'logistic_reg')."
            )

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
        """
        🔮 Generates predictions on unseen data.
        
        Loads a new CSV file and automatically pushes it through the exact 
        preprocessing graph (imputers, scalers, encoders) that was fit during 
        training, before calling `.predict()` on the underlying model.
        
        Args:
            new_data_csv_path (str): Filepath to the raw CSV data to predict on.
            
        Returns:
            np.ndarray: An array of predictions (floats for Regressors, 
                        classes/ints for Classifiers).
                        
        Raises:
            Exception: If the pipeline has not been trained yet.
        """
        if self.pipeline is None:
            raise Exception("Model not trained. Call train() first.")
            
        new_df = pd.read_csv(new_data_csv_path)
        return self.pipeline.predict(new_df)

    def predict_and_save(
        self, 
        test_csv_path: str, 
        output_csv_path: str, 
        output_target_col: str = None,
        keep_cols=None,
        transform_func: callable = None
    ):
        """
        🚀 Runs predictions on a test CSV and saves the results to a new file.
        
        Loads the test data, generates predictions, appends them as a new 
        column to the data, and saves the combined result to `output_csv_path`.
        
        Args:
            test_csv_path     (str): Filepath to the raw CSV data to predict on.
            output_csv_path   (str): Destination filepath for the resulting CSV.
            output_target_col (str): The name to give the new prediction column. 
                                     If None, defaults to the original target_col 
                                     used during training.
            keep_cols (str or list): If provided, ONLY these columns and the prediction 
                                     column will be saved.
            transform_func (callable): A function applied to predictions before saving 
                                       (e.g., np.expm1 for log-transformed targets).
        """
        print(f"Generating predictions for {test_csv_path}...")
        
        # Load the test data
        test_df = pd.read_csv(test_csv_path)
        
        # Get predictions (using the existing predict method)
        predictions = self.predict(test_csv_path)
        
        # Apply transformation if provided
        if transform_func is not None:
            predictions = transform_func(predictions)
        
        # Determine the column name for the predictions
        col_name = output_target_col if output_target_col else self.target_col
        if not col_name:
            col_name = 'Prediction' # Fallback if target_col wasn't set somehow
            
        # Add predictions to the dataframe
        test_df[col_name] = predictions

        # Filter to just keep_cols and Prediction if requested
        if keep_cols is not None:
            if isinstance(keep_cols, str):
                keep_cols = [keep_cols]
                
            missing_cols = [c for c in keep_cols if c not in test_df.columns]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in {test_csv_path}")
                
            save_df = test_df[list(keep_cols) + [col_name]]
        else:
            save_df = test_df
        
        # Save to the new CSV file
        import os
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        save_df.to_csv(output_csv_path, index=False)
        print(f"✅ Predictions saved successfully to {output_csv_path}")

    def save_model(self, filepath: str):
        """
        💾 Serializes the entire trained pipeline to disk securely.
        
        Automatically creates any missing parent directories in the filepath 
        and dumps the complete scikit-learn Pipeline (preprocessing + model) 
        using `joblib`.
        
        Args:
            filepath (str): The desired save path (e.g., 'saved_models/my_model.joblib').
        """
        if self.pipeline is None:
            raise Exception("No model to save.")
            
        import os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved securely to {filepath}")
        
    def load_model(self, filepath: str):
        """
        📂 Loads a previously saved pipeline from disk.
        
        Restores the full state of the model and preprocessors, attaching 
        them back to this `QuickModel` instance for immediate `.predict()` use.
        
        Args:
            filepath (str): Path to the existing .joblib file.
        """
        self.pipeline = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

    def run(self, csv_path: str, target_col: str, save_path: str, max_samples: int = None) -> 'QuickModel':
        """
        ✨ Super-function: End-to-end model creation in a single, fluent call.
        
        This method initializes, builds, trains on *csv_path*, and saves the 
        final artifact to *save_path*. It returns `self`, enabling method chaining.

        Args:
            csv_path    (str): Path to the historical training dataset (.csv).
            target_col  (str): Name of the column to predict.
            save_path   (str): Destination filepath for the saved .joblib model 
                               (e.g., 'models/revenue_model.joblib').
            max_samples (int): Max number of rows to train on.

        Returns:
            QuickModel: The trained ML engine instance (ready for .predict()).

        Example::

            # Initialize and run via instance
            model = QuickModel(model_type='binary_clf', nan_strategy='median')
            model.run('data/churn.csv', target_col='client_lost', save_path='models/churn.joblib')
        """
        self.train(csv_path=csv_path, target_col=target_col, max_samples=max_samples)
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
    max_samples: int = None,
) -> QuickModel:
    """
    ===========================================================================
    🎁 run_quick_model: The One-Liner Wrapper
    ===========================================================================
    
    A functional interface for the fastest possible path to a trained model.
    Initializes a QuickModel, trains it, and saves it—all in a single function call.

    Args:
        csv_path     (str): Path to the training CSV file.
        target_col   (str): Name of the column to predict.
        save_path    (str): Destination path for the saved model artifact.
        model_type   (str): Algorithm to use (default: 'random_forest_reg').
        nan_strategy (str): Missing-value strategy (default: 'median').
        max_samples  (int): Max number of rows to train on. Selects randomly if exceeded.

    Returns:
        QuickModel: The fully trained instance.

    Example::

        from src import run_quick_model

        model = run_quick_model(
            csv_path='data/my_data.csv',
            target_col='revenue',
            save_path='models/revenue_model.joblib',
            max_samples=10000,
        )
    """
    model = QuickModel(model_type=model_type, nan_strategy=nan_strategy)
    return model.run(csv_path=csv_path, target_col=target_col, save_path=save_path, max_samples=max_samples)