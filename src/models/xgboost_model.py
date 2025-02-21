import logging
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class XGBoostModel:
    """
    This class encapsulates a native XGBoost training pipeline using xgb.train (instead of the Sklearn API).
    It provides methods for training, predicting, evaluating, and extracting feature importance.
    """

    def __init__(self):
        """
        Initializes default hyperparameters for the native XGBoost API (xgb.train).

        Key hyperparameters:
        - booster: Which booster to use; "gbtree" is standard for tree-based models.
        - eta: Learning rate (smaller values lead to smoother, slower training).
        - max_depth: Maximum depth of each tree (shallower trees reduce overfitting).
        - min_child_weight: Minimum sum of instance weights (hessian) needed in a child node.
        - subsample: Fraction of training samples used for each boosting round.
        - colsample_bytree: Fraction of features used for constructing each tree.
        - gamma: Minimum loss reduction required for a split (lower value allows more splits).
        - lambda: L2 regularization term (corresponds to reg_lambda).
        - eval_metric: The evaluation metric; here "rmse" for regression tasks.
        - tree_method: Set to 'hist' to use the CPU histogram algorithm (ensures reliable booster extraction).
        - seed: Random seed for reproducibility.
        """
        self.params = {
            "booster": "gbtree",
            "eta": 0.05,                # Learning rate (lower values → smoother convergence)
            "max_depth": 6,             # Maximum depth of trees (shallower trees reduce overfitting)
            "min_child_weight": 1,      # Allow smaller leaves
            "subsample": 0.8,           # Use 80% of the data per boosting round
            "colsample_bytree": 0.8,    # Use 80% of the features per tree
            "gamma": 0,                 # Lower gamma means less conservative splits
            "lambda": 1.0,              # L2 regularization strength
            "eval_metric": "rmse",      # Evaluation metric for regression
            "tree_method": "hist",      # Use histogram-based algorithm (CPU)
            "seed": 42                  # Ensures reproducibility
        }
        # Number of boosting rounds (trees)
        self.num_boost_round = 1000

        # This will store the trained Booster model after training
        self.booster = None

        # This DataFrame will hold the extracted feature importance data after training
        self.importance_df = None

        # Additional storage to keep the last non-empty feature importance DataFrame
        self.last_importance_df = None

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50, verbose=0):
        """
        Trains the XGBoost model using xgb.train on DMatrix inputs. If validation data is provided,
        early stopping is attempted (if supported by the API). After training, feature importances are extracted.

        Parameters:
        - X_train (pd.DataFrame or np.array): Training features.
        - y_train (pd.Series or np.array): Training target values.
        - X_val (pd.DataFrame or np.array, optional): Validation features.
        - y_val (pd.Series or np.array, optional): Validation target values.
        - early_stopping_rounds (int, optional): Number of rounds with no improvement on the validation set before stopping.
        - verbose (int, optional): Frequency (in rounds) for logging the training progress.
        """
        logging.info("Training XGBoost model using the native DMatrix API...")

        # Convert training data into a DMatrix format, explicitly defining feature names if using a DataFrame.
        if isinstance(X_train, pd.DataFrame):
            dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=list(X_train.columns))
        else:
            dtrain = xgb.DMatrix(data=X_train, label=y_train)

        # Prepare the evaluation dataset list, always including training data.
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=list(X_val.columns))
            else:
                dval = xgb.DMatrix(data=X_val, label=y_val)
            evals.append((dval, "eval"))

        # Train the model, attempting early stopping if supported.
        try:
            self.booster = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose
            )
        except TypeError as e:
            logging.warning(f"Early stopping not supported or error encountered: {e}")
            logging.warning("Retrying training without early_stopping_rounds.")
            self.booster = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
                evals=evals,
                verbose_eval=verbose
            )

        logging.info("XGBoost training completed. Extracting feature importance...")

        # --- Feature Importance Extraction ---
        # Define feature names based on input type.
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

        # # Detailed log output (e.g. about feature names used and intermediate results) is only logged at DEBUG level.
        logging.debug(f"Using feature names: {feature_names}")

        importance = {}
        try:
            logging.debug("Attempting to extract feature importance using importance_type='gain'")
            importance = self.booster.get_score(importance_type="gain")
            logging.debug(f"Importance (gain): {importance}")
            if not importance:
                logging.debug("No importance data from 'gain'. Trying 'weight'.")
                importance = self.booster.get_score(importance_type="weight")
                logging.debug(f"Importance (weight): {importance}")
            if not importance:
                logging.debug("No importance data from 'weight'. Trying 'cover'.")
                importance = self.booster.get_score(importance_type="cover")
                logging.debug(f"Importance (cover): {importance}")
        except Exception as e:
            logging.warning(f"Could not extract feature importance via get_score(): {e}")
            importance = {}

        # Process and map importance values to feature names.
        if importance:
            sorted_items = []
            for k, v in importance.items():
                # Extract numerical index from keys like 'f0', 'f1', etc., and map to feature names.
                if k.startswith("f") and k[1:].isdigit():
                    try:
                        idx = int(k[1:])
                        fname = feature_names[idx] if idx < len(feature_names) else k
                    except ValueError:
                        fname = k
                else:
                    fname = k
                sorted_items.append((fname, v))
            # Create a sorted DataFrame with feature importance values.
            self.importance_df = pd.DataFrame(sorted_items, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
            self.last_importance_df = self.importance_df.copy()  # Backup the last extracted importance.

            # Zusammenfassende INFO-Ausgabe: Es werden nur die Top 3 wichtigen Features ausgegeben.
            logging.info(f"Top 3 important features: {self.importance_df.head(3).to_dict()}")
        else:
            logging.warning("Feature importance could not be retrieved; setting all importances to 0.")
            self.importance_df = pd.DataFrame({"Feature": feature_names, "Importance": [0] * len(feature_names)})
            if self.last_importance_df is None:
                self.last_importance_df = self.importance_df.copy()

    def predict(self, X_test):
        """
        Generates predictions using the trained Booster model.

        Parameters:
        - X_test (pd.DataFrame or np.array): Testing features.

        Returns:
        - np.array: The predicted target values.
        """
        if self.booster is None:
            logging.error("Booster not found. Please call train() before predict().")
            return None
        logging.info("Generating predictions with XGBoost booster...")
        if isinstance(X_test, pd.DataFrame):
            dtest = xgb.DMatrix(data=X_test, feature_names=list(X_test.columns))
        else:
            dtest = xgb.DMatrix(data=X_test)
        return self.booster.predict(dtest)

    def get_metrics(self, X_train, y_train, X_test, y_test):
        """
        Computes evaluation metrics for the trained XGBoost booster on both training and test data.

        Metrics:
        - MSE: Mean Squared Error.
        - RMSE: Root Mean Squared Error.
        - MAE: Mean Absolute Error.
        - R²: Coefficient of Determination.

        Parameters:
        - X_train, y_train: Training data.
        - X_test, y_test: Test data.

        Returns:
        - dict: A dictionary containing computed metrics.
        """
        logging.info("Calculating evaluation metrics for XGBoost booster...")

        if self.booster is None:
            logging.error("Booster not found. Please call train() first.")
            return {}

        if isinstance(X_train, pd.DataFrame):
            dtrain = xgb.DMatrix(data=X_train, feature_names=list(X_train.columns))
        else:
            dtrain = xgb.DMatrix(data=X_train)
        train_preds = self.booster.predict(dtrain)

        if isinstance(X_test, pd.DataFrame):
            dtest = xgb.DMatrix(data=X_test, feature_names=list(X_test.columns))
        else:
            dtest = xgb.DMatrix(data=X_test)
        test_preds = self.booster.predict(dtest)

        metrics = {
            "train-mse": mean_squared_error(y_train, train_preds),
            "test-mse": mean_squared_error(y_test, test_preds),
            "train-rmse": np.sqrt(mean_squared_error(y_train, train_preds)),
            "test-rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
            "train-mae": mean_absolute_error(y_train, train_preds),
            "test-mae": mean_absolute_error(y_test, test_preds),
            "train-r2": r2_score(y_train, train_preds),
            "test-r2": r2_score(y_test, test_preds)
        }
        logging.info(f"Training Metrics: {metrics}")
        logging.info(f"Testing Metrics: {metrics}")
        return metrics

    def get_feature_importance(self):
        """
        Retrieves and returns the feature importance scores based on the trained XGBoost booster.
        
        Returns:
            pd.DataFrame: A DataFrame containing feature names and their corresponding importance scores,
                        sorted in descending order. If no importance data is available, returns an empty DataFrame.
        
        Detailed Explanation:
        1. This method first checks whether the current feature importance DataFrame (self.importance_df)
           exists and is not empty. This DataFrame is populated during the training process when the feature
           importances are extracted using the Booster's get_score() method.
        2. If self.importance_df is available and non-empty, it is sorted in descending order by the 'Importance'
           column. This ensures that the most important features appear at the top of the DataFrame.
        3. If self.importance_df is empty, the method then checks for a backup copy stored in 
           self.last_importance_df. This backup holds the last non-empty feature importance DataFrame obtained
           during a previous training run.
        4. If the backup exists and is non-empty, a log message is recorded and that backup DataFrame is returned,
           again sorted in descending order.
        5. If neither the current nor the backup DataFrame is available, a warning is logged, and an empty 
           DataFrame with the expected columns is returned.
        
        Note: This method does not generate any plots; it only provides the raw importance data.
        """
        # First, check if the current feature importance DataFrame exists and is not empty.
        if self.importance_df is not None and not self.importance_df.empty:
            return self.importance_df.sort_values(by="Importance", ascending=False)
        # If not, check if a backup of the last non-empty importance DataFrame exists.
        elif self.last_importance_df is not None and not self.last_importance_df.empty:
            logging.info("Returning last non-empty feature importance data.")
            return self.last_importance_df.sort_values(by="Importance", ascending=False)
        # If no data is available, log a warning and return an empty DataFrame.
        else:
            logging.warning("No feature importance available. Train the model first or check logs for warnings.")
            return pd.DataFrame(columns=["Feature", "Importance"])
        
    def get_booster(self):
        """
        Returns the underlying XGBoost Booster object.
        This allows external code (e.g. in ModelEvaluation or ModelComparator)
        to retrieve the Booster for additional operations.
        """
        return self.booster