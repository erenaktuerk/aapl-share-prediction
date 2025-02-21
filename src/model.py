import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel
from src.models.model_comparator import ModelComparator

class StockPricePredictor:
    """
    This class manages the complete stock price prediction workflow using:
      - XGBoost (Gradient Boosting)
      - MLP (Multi-Layer Perceptron - Neural Network)

    The workflow includes:
      1. Loading and preparing the dataset (features and target)
      2. Splitting the dataset into training and test sets
      3. Training both models
      4. Evaluating model performance
      5. Comparing models to determine the best one
    """

    def __init__(self, data):
        """
        Initializes the StockPricePredictor by loading and preparing the dataset.

        Parameters:
        - data (str or pd.DataFrame): Dataset file path (CSV) or preloaded pandas DataFrame.
        """
        logging.info("Initializing StockPricePredictor...")

        # Load the dataset if a file path is passed
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data.copy()  # Copy the DataFrame to avoid modifying the original
        else:
            raise ValueError("Data must be a file path (str) or a pandas DataFrame.")

        # Define the features (X) and the target (y)
        # If a "Date" column exists, it is dropped along with "Close Price"
        if "Date" in self.data.columns:
            self.X = self.data.drop(columns=["Date", "Close Price"], errors="ignore")
        else:
            self.X = self.data.drop(columns=["Close Price"], errors="ignore")
        self.y = self.data["Close Price"]

        # Placeholder for the train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # Initialize model instances
        self.xgb_model = XGBoostModel()
        self.mlp_model = None  # This will be initialized later after input dimensions are known

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.

        Parameters:
        - test_size (float): Proportion of dataset for testing.
        - random_state (int): Controls shuffling before splitting.
        """
        logging.info("Splitting dataset into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

    def train_models(self):
        """
        Trains both the XGBoost and MLP models on the training dataset.
        
        This method performs the following steps:
        1. Suppresses specific Keras warnings regarding the use of 'input_shape' or 'input_dim'
            in layer initialization. These warnings are non-critical and known, so we filter them out
            to reduce unnecessary log clutter.
        2. Verifies that the training data has been split into training and testing sets.
        3. Initializes the XGBoost model instance.
        4. Initializes the MLP model instance, setting its input dimension based on the number of features.
        5. Trains the XGBoost model using the training set and passes the test set as validation data.
            The verbosity is set to 0 to minimize periodic log output from the XGBoost training process.
        6. Trains the MLP model using only the training data.
        
        Raises:
        RuntimeError: If the training data (X_train or y_train) is not available.
        """
        # Suppress Keras warnings related to layer initialization that are known and non-critical.
        warnings.filterwarnings(
            "ignore",
            message="Do not pass an input_shape/input_dim argument to a layer",
            module="keras.src.layers.core.dense"
        )
        
        # Log the start of the model training process.
        logging.info("Training XGBoost and MLP models...")

        # Ensure that the training data has been set (i.e., split_data() has been called).
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Data must be split before training. Call split_data() first.")

        # Initialize the XGBoost model instance.
        self.xgb_model = XGBoostModel()

        # Initialize the MLP model instance with the appropriate input dimension,
        # which is determined by the number of columns in the training feature set.
        self.mlp_model = MLPModel(input_dim=self.X_train.shape[1])

        # Train the XGBoost model:
        # - Provide training features and labels from self.X_train and self.y_train.
        # - Pass the test set (self.X_test, self.y_test) as validation data to monitor performance.
        # - Set the 'verbose' parameter to 0 to suppress unnecessary periodic logging during training.
        self.xgb_model.train(self.X_train, self.y_train, self.X_test, self.y_test, verbose=50)

        # Train the MLP model using only the training data.
        # It is assumed that the MLP model's train() method handles its internal logging appropriately.
        self.mlp_model.train(self.X_train, self.y_train)

    def evaluate_models(self):
        """
        Evaluates both trained models using multiple performance metrics.

        Returns:
        - tuple: (xgb_metrics, mlp_metrics), where each is a dictionary of evaluation metrics.
        """
        logging.info("Evaluating model performance...")
        xgb_metrics = self.xgb_model.get_metrics(self.X_train, self.y_train, self.X_test, self.y_test)
        mlp_metrics = self.mlp_model.get_metrics(self.X_train, self.y_train, self.X_test, self.y_test)
        return xgb_metrics, mlp_metrics

    def determine_best_model(self):
        """
        Determines the best model based on evaluation metrics by comparing the results.

        Returns:
        - tuple: (best_model_name, best_metrics), where best_model_name is a string ("XGBoost" or "MLP")
                 and best_metrics is a dictionary of the evaluation metrics for the best model.
        """
        logging.info("Determining the best model...")

        # Split the data if it hasn't been split already
        self.split_data()

        # Train both models on the training data
        self.train_models()

        # Evaluate both models
        xgb_metrics, mlp_metrics = self.evaluate_models()
        logging.info(f"XGBoost Metrics: {xgb_metrics}")
        logging.info(f"MLP Metrics: {mlp_metrics}")

        # Compare the models using ModelComparator
        comparator = ModelComparator(xgb_metrics, mlp_metrics)
        best_model, best_metrics = comparator.compare_models()

        logging.info(f"Best Model: {best_model}")
        logging.info(f"Best Model Metrics: {best_metrics}")