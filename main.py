import logging
import os
import pandas as pd
from src.data_preprocessing import DataPreprocessing
from src.model import StockPricePredictor
from src.evaluation import ModelEvaluation
from src.models.xgboost_model import XGBoostModel
from src.models.mlp_model import MLPModel
from src.models.model_comparator import ModelComparator

# Configure logging to display information messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress warnings


def main():
    """
    Main pipeline for the stock price prediction process:
      1. Data preprocessing
      2. Splitting the dataset into training and test sets
      3. Training both models (XGBoost and MLP)
      4. Evaluating model performance
      5. Comparing models to determine the best one
      6. Visualizing the results for the best model
    """
    logging.info("Starting the stock price prediction pipeline...")

    # --- Step 1: Data Preprocessing ---
    # Create a DataPreprocessing object with the input and output file paths.
    preprocessor = DataPreprocessing(
        input_file="data/raw/raw_share_data.csv",
        output_file="data/processed/processed_share_data.csv"
    )
    # Execute the full preprocessing process: load data, clean it, engineer features, and save.
    preprocessor.preprocess_data()

    # Load the processed data from the output file.
    processed_data = pd.read_csv(preprocessor.output_file)
    logging.info("Processed data successfully loaded.")

    # --- Step 2: Initialize StockPricePredictor ---
    # The StockPricePredictor extracts features (X) and the target (y) from the DataFrame.
    predictor = StockPricePredictor(processed_data)

    # --- Step 3: Split the Data ---
    # Split the dataset into training and test sets (e.g., 80% training, 20% testing).
    predictor.split_data(test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    # --- Step 4: Train the Models ---
    # Call the internal method to build and train both the XGBoost and MLP models.
    predictor.train_models()
    logging.info("Both models trained successfully.")

    # --- Step 5: Evaluate the Models ---
    # Evaluate both models and obtain their performance metrics.
    xgb_metrics, mlp_metrics = predictor.evaluate_models()
    logging.info(f"XGBoost Metrics: {xgb_metrics}")
    logging.info(f"MLP Metrics: {mlp_metrics}")

    # --- Step 6: Compare the Models ---
    # Instantiate the ModelComparator with the evaluation metrics and the trained model instances.
    comparator = ModelComparator(xgb_metrics, mlp_metrics,
                                 mlp_model=predictor.mlp_model,
                                 xgboost_model=predictor.xgb_model)
    try:
        best_model_name, best_model_metrics = comparator.compare_models()
        logging.info(f"Best Model: {best_model_name}")
        logging.info(f"Best Model Metrics: {best_model_metrics}")
    except KeyError as e:
        logging.error(f"Error comparing models: {e}")
        return

    # --- Step 7: Visualize the Results for the Best Model ---
    # Select the best model based on the comparison result.
    best_model = predictor.xgb_model if best_model_name == "XGBoost" else predictor.mlp_model

    # Generate predictions for the test set.
    if best_model_name == "XGBoost":
        y_pred = best_model.predict(predictor.X_test)
    else:
        # For the MLP model, flatten the predictions (which may be returned as a 2D array).
        y_pred = best_model.predict(predictor.X_test).flatten()

    # Create a ModelEvaluation object using the test labels, predictions, and the best model.
    evaluator = ModelEvaluation(predictor.y_test, y_pred, best_model, best_model,
                                predictor.X_train, predictor.y_train)
    # Run the evaluation and plot all results (metrics, actual vs. predicted, residuals, etc.).
    evaluator.evaluate_all()

    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()