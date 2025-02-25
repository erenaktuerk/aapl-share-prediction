AAPL Share Price Prediction using Machine Learning

📌 Overview

This project focuses on predicting the stock prices of Apple Inc. (AAPL) using machine learning techniques. With a strong emphasis on real-world application, the project leverages two state-of-the-art models: XGBoost and a Multi-Layer Perceptron (MLP) implemented in TensorFlow. The model comparison is automated, ensuring that the best-performing model is selected for further evaluation.

This project showcases a highly optimized pipeline with a focus on data preprocessing, feature engineering, hyperparameter tuning, and model evaluation. It is designed to impress companies looking for practical, production-ready machine learning solutions.

🚀 Key Features

✔ Data Preprocessing: Clean, transform, and normalize stock market data for machine learning applications.

✔ Feature Engineering: Identify and select the most impactful features for improved prediction accuracy.

✔ Model Selection: Automated comparison between XGBoost and MLP to choose the best regression model.

✔ Hyperparameter Optimization: RandomizedSearchCV to fine-tune the hyperparameters of the models.

✔ Cross-Validation: Implement robust k-fold cross-validation to improve model generalization.

✔ Feature Importance Analysis: Understand the key factors affecting stock price predictions.

✔ Automated Logging & Debugging: Clear terminal output with detailed logging for efficient debugging.

✔ Performance Visualization: Generate insightful visualizations to assess model performance.

✔ Professional Code Structure: Modularized code for scalability and maintainability.

📂 Project Structure

aapl-share-prediction/
├── .gitignore
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   │   └── processed_share_data.csv
│   └── raw/
│       └── raw_share_data.csv
├── src/
│   ├── _init_.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── model.py
│   └── models/
│       ├── mlp_model.py
│       ├── model_comparator.py
│       └── xgboost_model.py
└── venv/

🔧 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/erenaktuerk/aapl-share-prediction.git
cd aapl-share-prediction

2️⃣ Create and Activate a Virtual Environment

To avoid dependency conflicts, it’s recommended to use a virtual environment:

For Windows (Command Prompt / PowerShell):

python -m venv venv
venv\Scripts\activate

For macOS / Linux:

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies

After activating the virtual environment, install the necessary packages:

pip install -r requirements.txt

4️⃣ Run the Complete Pipeline

To run the project and start the pipeline:

python main.py

This will:

	•	Preprocess the data
 
	•	Train and optimize the models
 
	•	Automatically select the best model
 
	•	Generate performance metrics and visualizations

📊 Data Preprocessing

The dataset used consists of Apple Inc. (AAPL) stock prices over a 5-year period. The following preprocessing steps are performed:

✔ Handling Missing Values: Ensures data consistency.
✔ Feature Selection: Only relevant columns like Date, Open, High, Low, Close, and Volume are kept.
✔ Normalization & Scaling: Ensures the model converges effectively.
✔ Train-Test Split: Splits the data for reliable evaluation.

📈 Machine Learning Models

1️⃣ XGBoost Regressor

XGBoost is a powerful gradient boosting algorithm known for its performance and scalability.
	•	Key Features:
	•	Efficient gradient boosting for regression tasks.
	•	Early stopping and feature importance analysis.
	•	Hyperparameter tuning using RandomizedSearchCV.

Key Parameters Tuned:
	•	learning_rate
	•	n_estimators
	•	max_depth
	•	gamma
	•	min_child_weight
	•	reg_lambda

2️⃣ Multi-Layer Perceptron (MLP) with TensorFlow

This deep learning model is implemented using tf.keras, a high-level neural networks API.
	•	Architecture:
	•	Input Layer: Features from the dataset.
	•	Hidden Layers: 2-3 layers with Dropout for regularization.
	•	Output Layer: Single neuron for predicting stock prices.
	•	Loss Function: Mean Squared Error (MSE) for regression tasks.

🔬 Model Evaluation & Selection

Each model is evaluated based on the following metrics:

✔ Root Mean Squared Error (RMSE)
✔ Mean Absolute Error (MAE)
✔ R² Score

The model with the lowest RMSE and best performance metrics is automatically selected.

Performance Visualizations:
	•	Actual vs. Predicted Prices: To visualize the model’s prediction accuracy.
	•	Error Distribution Plot: To visualize the errors in the predictions.
	•	Feature Importance Bar Chart: To understand which features impact the stock price predictions.

🎯 Why This Project Stands Out

This project is a comprehensive, end-to-end machine learning solution tailored to predict stock prices with both traditional regression and deep learning methods.

Why it’s impressive:
	•	Practical application: Stock price forecasting has significant real-world value, especially in financial analytics.
	•	Advanced Techniques: XGBoost and MLP represent modern machine learning techniques. The use of RandomizedSearchCV and cross-validation ensures optimal model performance.
	•	Feature Engineering: Thoughtful feature selection and normalization ensure that the data is ready for optimal model performance.
	•	Code Modularity: The well-structured, modularized code is designed for scalability and maintainability in production environments.

🛠 Logging & Debugging

Efficient logging ensures that users can debug the pipeline without cluttering the terminal output:
	•	INFO level logs: Clean, user-friendly terminal output.
	•	DEBUG level logs: Stored in logs/ directory for detailed analysis.

💡 Future Enhancements

Future improvements for the project could include:
	•	LSTM for Time-Series Forecasting: Integrate Long Short-Term Memory networks for more accurate predictions on time-series data.
	•	Sentiment Analysis: Include sentiment analysis from financial news to potentially improve stock price predictions.
	•	Model Deployment: Deploy the model using a REST API to provide real-time stock predictions.
	•	Model Optimization: Further hyperparameter tuning and testing of additional models.

📜 Lessons Learned

This project not only demonstrates the capability to work with advanced regression models and deep learning but also the importance of effective data preprocessing, feature engineering, and model optimization.

Key Takeaways:
	•	Hyperparameter tuning is critical for improving model performance.
	•	Data quality and preprocessing directly influence prediction accuracy.
	•	Continuous model evaluation and improvement are necessary for high-performance systems.

📩 Contact & Contributions

For any questions, suggestions, or improvements, feel free to reach out:

📧 erenaktuerk@hotmail.com
🌐 github.com/erenaktuerk

Want to contribute? Fork the repository, create a feature branch, and submit a pull request! 🚀
