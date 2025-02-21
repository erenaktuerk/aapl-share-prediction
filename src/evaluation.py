import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, train_test_split
from src.models.xgboost_model import XGBoostModel
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configure seaborn style for clean and modern plots
sns.set(style="whitegrid")


class ModelEvaluation:
    """
    The ModelEvaluation class provides a comprehensive evaluation of a regression model.
    It includes performance metrics, feature importance, residual analysis, actual vs. predicted plots,
    error distribution, and a learning curve (if applicable).
    """
    def __init__(self, y_test, y_pred, importance_df, model, X_train, y_train, X_val=None, y_val=None):
        """
        Initialize the evaluation class with test labels, predictions, feature importance, and training data.

        Parameters:
            y_test (array-like): True target values of the test set.
            y_pred (array-like): Predicted target values.
            importance_df (DataFrame): DataFrame containing feature importance.
            model: The trained model (XGBoostModel or MLP).
            X_train (DataFrame): Training feature data.
            y_train (array-like): Training target values.
            X_val (DataFrame, optional): Validation feature data. Default is None.
            y_val (array-like, optional): Validation target values. Default is None.
        """
        if len(y_test) != len(y_pred):
            raise ValueError("Mismatch between y_test and y_pred: they must have the same length.")
        self.y_test = np.array(y_test)
        self.y_pred = np.array(y_pred)
        # We store the feature importance passed at initialization, but note that we now
        # prefer to call the model's get_feature_importance() method to always retrieve the latest values.
        self.importance_df = importance_df
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def print_metrics(self):
        """
        Calculate and print common regression evaluation metrics:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R² Score
        """
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        print("\nModel Performance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

    def plot_learning_curve(self, title="Learning Curve"):
        """
        Plot the learning curve to illustrate how model performance changes with the size of the training set.
        This is only applicable for models compatible with sklearn's learning_curve function.
        For XGBoost models, if validation data is not provided, a validation set is created automatically
        from the training data (using an 80/20 split).

        Detailed Steps:
        - If the model is a TensorFlow model, print that learning curves are not supported.
        - If the model is an instance of XGBoostModel:
            - Check whether validation data (X_val, y_val) is provided.
            - If not provided, automatically split self.X_train and self.y_train (80/20 split).
            - Then, for a fixed number (e.g., 10) of training set sizes, fit the model on the subset
              (using the train() method) and compute training and validation errors.
              Here, we set verbose=False to reduce output spam.
            - Plot these errors against the training set sizes.
        - Otherwise, for Scikit-Learn–compatible models, use sklearn’s learning_curve() function.
        """
        if isinstance(self.model, tf.keras.Model):
            print("Learning curve is not supported for TensorFlow models.")
            return

        try:
            # Special handling for XGBoostModel
            if isinstance(self.model, XGBoostModel):
                # If validation data is not provided, automatically create a hold-out set (20%).
                if self.X_val is None or self.y_val is None:
                    print("No validation data provided for XGBoost learning curve; creating a validation set (20% hold-out).")
                    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                        self.X_train, self.y_train, test_size=0.2, random_state=42
                    )
                else:
                    X_train_sub = self.X_train.copy()
                    y_train_sub = self.y_train.copy()
                    X_val = self.X_val
                    y_val = self.y_val

                print("Generating learning curve for XGBoost model...")

                # Define a fixed number of training sizes to evaluate (e.g., 10 evenly spaced sizes)
                n_samples = len(X_train_sub)
                min_samples = min(50, n_samples)
                training_sizes = np.linspace(min_samples, n_samples, num=10, dtype=int)

                train_errors = []
                val_errors = []

                # Loop over the selected training sizes
                for size in training_sizes:
                    # Select a subset of the training data of the given size
                    X_subset = X_train_sub[:size]
                    y_subset = self.y_train[:size]

                    # Train the model on the current subset using the train() method.
                    # Set verbose=False to suppress detailed training logs.
                    self.model.train(X_subset, y_subset, X_val, y_val, early_stopping_rounds=10, verbose=False)

                    # Calculate training error (MSE) for the current subset
                    train_pred = self.model.predict(X_subset)
                    train_error = np.mean((train_pred - y_subset) ** 2)
                    train_errors.append(train_error)

                    # Calculate validation error (MSE) on the fixed validation set
                    val_pred = self.model.predict(X_val)
                    val_error = np.mean((val_pred - y_val) ** 2)
                    val_errors.append(val_error)

                # Plot the computed learning curve
                plt.figure(figsize=(10, 6))
                plt.plot(training_sizes, train_errors, 'o-', color="blue", label="Training Error")
                plt.plot(training_sizes, val_errors, 'o-', color="red", label="Validation Error")
                plt.title(title)
                plt.xlabel("Training Set Size")
                plt.ylabel("Error (MSE)")
                plt.legend()
                plt.tight_layout()
                plt.show()

            else:
                # For Scikit-Learn compatible models, use the learning_curve function.
                estimator = self.model.model if isinstance(self.model, XGBoostModel) else self.model
                if hasattr(estimator, 'fit') and hasattr(estimator, 'predict'):
                    train_sizes, train_scores, test_scores = learning_curve(
                        estimator, self.X_train, self.y_train, cv=5,
                        scoring="neg_mean_squared_error", n_jobs=-1
                    )
                    # Convert negative MSE scores to positive values.
                    train_scores_mean = -np.mean(train_scores, axis=1)
                    test_scores_mean = -np.mean(test_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    test_scores_std = np.std(test_scores, axis=1)

                    plt.figure(figsize=(10, 6))
                    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Error")
                    plt.plot(train_sizes, test_scores_mean, 'o-', color="red", label="Validation Error")
                    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                     train_scores_mean + train_scores_std, alpha=0.2, color="blue")
                    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                     test_scores_mean + test_scores_std, alpha=0.2, color="red")
                    plt.title(title)
                    plt.xlabel("Training Set Size")
                    plt.ylabel("Error (MSE)")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                else:
                    print("Model is not a valid Scikit-Learn model for learning curve.")

        except Exception as e:
            print(f"Could not generate learning curve: {e}")

    def plot_feature_importance(self):
        """
        Plot a horizontal bar chart of the feature importances.
        This method uses the feature importance data extracted by the model.
        
        Note:
            Instead of directly using a stored importance_df, this method calls the model's
            get_feature_importance() method to always retrieve the latest feature importance data.
        """
        # Retrieve the most up-to-date feature importance data from the model.
        importance_df = self.model.get_feature_importance()
        
        # Check if the returned DataFrame is non-empty.
        if not (isinstance(importance_df, pd.DataFrame) and not importance_df.empty):
            print("No feature importance data available.")
            return

        # Sort the feature importance data by importance value in descending order.
        sorted_df = importance_df.sort_values(by="Importance", ascending=False)
        
        # Create a horizontal bar chart using seaborn.
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=sorted_df, palette="viridis", legend=False)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    def plot_residuals(self):
        """
        Plot the distribution of residuals to check for patterns that indicate model bias.
        A residual plot can show if errors are randomly distributed or if there is a systematic bias.
        """
        residuals = self.y_test - self.y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30, color='blue')
        plt.axvline(x=0, color='red', linestyle='dashed')
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_actual_vs_predicted(self):
        """
        Plot actual vs. predicted values to visualize how well the model's predictions align with reality.
        A perfect model would have predictions exactly along the diagonal line.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.7, color='green')
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()], color='red', lw=2, linestyle="--")
        plt.title("Actual vs Predicted Share Price")
        plt.xlabel("Actual Share Price")
        plt.ylabel("Predicted Share Price")
        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self):
        """
        Plot the distribution of absolute errors to analyze how large the errors tend to be.
        This helps in identifying if there are any extreme values or outliers.
        """
        errors = np.abs(self.y_test - self.y_pred)
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=30, color='purple')
        plt.title("Error Distribution")
        plt.xlabel("Absolute Error")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def evaluate_all(self):
        """
        Run all evaluation methods sequentially for a complete model assessment.
        This method covers:
        - Model performance metrics
        - Actual vs. predicted values plot
        - Residuals distribution plot
        - Error distribution plot
        - Feature importance plot (if available)
        - Learning curve plot (if applicable)
        """
        self.print_metrics()
        self.plot_actual_vs_predicted()
        self.plot_residuals()
        self.plot_error_distribution()
        self.plot_feature_importance()
        self.plot_learning_curve()