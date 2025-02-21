import logging

class ModelComparator:
    """
    This class compares multiple regression models based on key evaluation metrics.
    It determines which model performs better by comparing individual metrics such as:
      - R² (Coefficient of Determination): Higher values (closer to 1) are better.
      - MAE (Mean Absolute Error): Lower values are better.
      - MSE (Mean Squared Error): Lower values are better.
      - RMSE (Root Mean Squared Error): Lower values are better.
      
    RMSE is weighted highest by default because it penalizes large errors more heavily,
    which is particularly important in applications where large deviations are very costly.
    """

    def __init__(self, metrics_xgb, metrics_mlp, mlp_model=None, xgboost_model=None):
        """
        Initializes the comparator with evaluation metrics from both models.

        Args:
            metrics_xgb (dict): Evaluation metrics for the XGBoost model.
                Expected keys: "test-r2", "test-mae", "test-mse", "test-rmse".
            metrics_mlp (dict): Evaluation metrics for the MLP model.
                Expected keys: "test-r2", "test-mae", "test-mse", "test-rmse" (or fallback keys with underscores,
                e.g., "test_r2", "test_mae", "test_mse", "test_rmse").
            mlp_model (object, optional): Instance of the MLP model (if retraining is desired).
            xgboost_model (object, optional): Instance of the XGBoost model (for feature importance extraction).

        Raises:
            ValueError: If either metrics_xgb or metrics_mlp is not a dictionary.
        """
        # Validate that both metric inputs are dictionaries.
        if not isinstance(metrics_xgb, dict) or not isinstance(metrics_mlp, dict):
            raise ValueError("Both model metrics should be dictionaries.")

        self.metrics_xgb = metrics_xgb
        self.metrics_mlp = metrics_mlp
        self.mlp_model = mlp_model
        self.xgboost_model = xgboost_model

    def get_metric(self, metrics, key):
        """
        Helper function to retrieve a metric value from a dictionary.
        It first attempts to retrieve the value using the provided key.
        If the key is not found, it replaces hyphens with underscores and tries again.

        Args:
            metrics (dict): Dictionary containing metric values.
            key (str): The key to search for (e.g., "test-r2").

        Returns:
            The metric value if found; otherwise, None.
        """
        value = metrics.get(key)
        if value is None:
            # Replace hyphens with underscores and try again.
            alt_key = key.replace("-", "_")
            value = metrics.get(alt_key)
        return value

    def compare_models(self, weights=None):
        """
        Compares the models based on multiple weighted evaluation metrics.

        The comparison is performed by assigning weights to each metric and then comparing
        the values for each model. For each metric, the model with the more favorable value
        (i.e., higher R², lower MAE, lower MSE, lower RMSE) receives one win.
        The overall best model is determined based on the win count.

        Args:
            weights (dict, optional): A dictionary with weights for each metric.
                Default weights are: {"RMSE": 0.4, "R2": 0.2, "MAE": 0.2, "MSE": 0.2}.
                RMSE is weighted highest because it penalizes large errors significantly.

        Returns:
            tuple: A tuple (best_model, best_metrics) where best_model is a string ("XGBoost" or "MLP")
                   and best_metrics is the corresponding dictionary of evaluation metrics.

        Raises:
            KeyError: If any required MLP metric is missing.
        """
        # Set default weights if none are provided.
        if weights is None:
            weights = {
                "RMSE": 0.4,  # High weight: large errors are especially undesirable.
                "R2": 0.2,
                "MAE": 0.2,
                "MSE": 0.2
            }

        # Normalize weights so that the sum equals 1.
        total_weight = sum(weights.values())
        weights = {key: value / total_weight for key, value in weights.items()}

        # Retrieve evaluation metrics for XGBoost using the helper function.
        r2_xgb = self.get_metric(self.metrics_xgb, "test-r2")
        mae_xgb = self.get_metric(self.metrics_xgb, "test-mae")
        mse_xgb = self.get_metric(self.metrics_xgb, "test-mse")
        rmse_xgb = self.get_metric(self.metrics_xgb, "test-rmse")

        # Retrieve evaluation metrics for MLP using the helper function.
        r2_mlp = self.get_metric(self.metrics_mlp, "test-r2")
        mae_mlp = self.get_metric(self.metrics_mlp, "test-mae")
        mse_mlp = self.get_metric(self.metrics_mlp, "test-mse")
        rmse_mlp = self.get_metric(self.metrics_mlp, "test-rmse")

        # Check for any missing MLP metrics.
        missing_metrics = [key for key, value in {
            "test-r2": r2_mlp,
            "test-mae": mae_mlp,
            "test-mse": mse_mlp,
            "test-rmse": rmse_mlp
        }.items() if value is None]
        if missing_metrics:
            raise KeyError(f"Missing required MLP metrics: {', '.join(missing_metrics)}")

        # Initialize counters for wins.
        xgb_wins = 0
        mlp_wins = 0

        # Compare R²: Higher is better.
        if r2_xgb > r2_mlp:
            xgb_wins += 1
        else:
            mlp_wins += 1

        # Compare MAE: Lower is better.
        if mae_xgb < mae_mlp:
            xgb_wins += 1
        else:
            mlp_wins += 1

        # Compare MSE: Lower is better.
        if mse_xgb < mse_mlp:
            xgb_wins += 1
        else:
            mlp_wins += 1

        # Compare RMSE: Lower is better.
        if rmse_xgb < rmse_mlp:
            xgb_wins += 1
        else:
            mlp_wins += 1

        logging.info(f"XGBoost wins count: {xgb_wins}, MLP wins count: {mlp_wins}")

        # Determine the overall best model.
        if xgb_wins > mlp_wins:
            best_model = "XGBoost"
            best_metrics = self.metrics_xgb
        else:
            best_model = "MLP"
            best_metrics = self.metrics_mlp

        logging.info(f"Best model selected: {best_model}")

        # If the best model is XGBoost and an instance is provided, attempt to extract feature importances.
        if best_model == "XGBoost" and self.xgboost_model is not None:
            logging.info("Extracting feature importances for XGBoost.")
            try:
                # Use get_booster() to access the booster and then get the score.
                feature_importance = self.xgboost_model.get_booster().get_score(importance_type='weight')
                logging.info(f"Feature importances for XGBoost: {feature_importance}")
            except AttributeError as e:
                logging.warning(f"Could not extract feature importances: {e}")

        # Optionally, if MLP is selected and retraining is supported, trigger retraining.
        if best_model == "MLP" and self.mlp_model is not None:
            logging.info("Retraining MLP model as it was selected as the best model.")
            if hasattr(self.mlp_model, "retrain_best_model"):
                self.mlp_model.retrain_best_model()
            else:
                logging.warning("MLP model does not support retraining.")

        return best_model, best_metrics