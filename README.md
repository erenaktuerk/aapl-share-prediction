AAPL Stock Price Prediction using Machine Learning

📌 Overview

This project is a highly optimized stock price prediction system that leverages advanced regression models to predict Apple Inc. (AAPL) stock prices. It is designed to showcase state-of-the-art machine learning techniques with a strong focus on practical implementation, performance tuning, and feature engineering.

The project compares XGBoost, a powerful gradient boosting algorithm, with a Multi-Layer Perceptron (MLP) neural network implemented in TensorFlow. The model selection is fully automated, ensuring that only the best-performing model is used for evaluation and visualization.

This work is crafted to demonstrate real-world expertise in data science and machine learning, making it an ideal portfolio project for a Machine Learning Engineer role.

🚀 Key Features

✔ Data Preprocessing: Clean and transform stock market data efficiently.
✔ Feature Engineering: Select the most impactful features for prediction.
✔ Model Selection: Compare and choose the best regression model (XGBoost vs. MLP).
✔ Hyperparameter Optimization: Use RandomizedSearchCV to fine-tune model parameters.
✔ Cross-Validation: Implement robust k-fold cross-validation for better generalization.
✔ Feature Importance Analysis: Identify key factors influencing stock prices.
✔ Automated Logging & Debugging: Keep terminal output clean while storing detailed logs.
✔ Performance Visualization: Generate insightful plots to analyze model predictions.
✔ Professional Code Structure: Modularized for scalability and maintainability.

📂 Project Structure

📦 aapl-stock-prediction
│── 📂 data
│   ├── 📂 raw                  # Original raw dataset (CSV format)
│   ├── 📂 processed            # Preprocessed and cleaned dataset
│
│── 📂 models                   
│   ├── xgboost_model.py        # XGBoost regression model implementation
│   ├── mlp_model.py            # TensorFlow MLP model implementation
│
│── 📂 evaluation               
│   ├── evaluation.py           # Model evaluation, performance metrics, and visualizations
│
│── 📂 venv                     # Virtual environment for package management
│
│── 📜 preprocess.py            # Data preprocessing pipeline
│── 📜 main.py                   # Main script to run the complete pipeline
│── 📜 README.md                 # Project documentation
│── 📜 requirements.txt          # Dependencies for easy setup

🔧 Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/erenaktuerk/aapl-stock-prediction.git
cd aapl-stock-prediction

2️⃣ Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to keep dependencies organized and avoid conflicts.

For Windows (Command Prompt / PowerShell)

python -m venv venv
venv\Scripts\activate

For macOS / Linux

python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies

Once the virtual environment is activated, install all necessary packages:

pip install -r requirements.txt

4️⃣ Run the Complete Pipeline

python main.py

This will:
✅ Preprocess the data
✅ Train and optimize the models
✅ Select the best model automatically
✅ Generate evaluation metrics and visualizations

📊 Data Preprocessing

The dataset consists of Apple Inc. stock price data over a 5-year period. The following steps are performed:

✔ Handling Missing Values: Ensuring data consistency.
✔ Feature Selection: Keeping only the most relevant columns (Date, Open, High, Low, Close, Volume).
✔ Normalization & Scaling: Improving model convergence.
✔ Train-Test Split: Ensuring reliable model evaluation.

📈 Machine Learning Models

1️⃣ XGBoost Regressor
	•	Implements gradient boosting to optimize stock price prediction.
	•	Highly efficient with early stopping and feature importance analysis.
	•	Hyperparameter tuning using RandomizedSearchCV.

Key Parameters Tuned:

✔ learning_rate
✔ n_estimators
✔ max_depth
✔ gamma
✔ min_child_weight
✔ reg_lambda

2️⃣ Multi-Layer Perceptron (MLP) with TensorFlow
	•	Implements a deep learning model using tf.keras.
	•	Fully connected neural network with multiple layers.
	•	Uses ReLU activation and Adam optimizer for stability.

Architecture:
	•	Input Layer: Number of selected features
	•	Hidden Layers: 2-3 layers with Dropout for regularization
	•	Output Layer: 1 neuron for regression output
	•	Loss Function: Mean Squared Error (MSE)

🔬 Model Evaluation & Selection

✅ Performance Metrics

Each model is evaluated based on:

✔ Root Mean Squared Error (RMSE)
✔ Mean Absolute Error (MAE)
✔ R² Score (Coefficient of Determination)

The best-performing model is automatically selected for visualization and final predictions.

📉 Data Visualization

The project generates various plots to analyze model performance:

✔ Actual vs. Predicted Prices
✔ Error Distribution Plot
✔ Feature Importance Bar Chart

These plots help in understanding how well the model captures stock price trends.

🛠 Logging & Debugging
	•	INFO level logs are printed for a clean terminal output.
	•	DEBUG level logs are saved in logs/ for detailed analysis.
	•	Ensures efficient debugging without cluttering the user experience.

🎯 Why This Project Stands Out

✔ Designed for real-world application in stock price forecasting.
✔ Uses advanced machine learning & deep learning techniques.
✔ Highly optimized feature selection & hyperparameter tuning.
✔ Scalable, well-documented, and modular code structure.
✔ Fully automated model selection and evaluation.
✔ Professional and production-ready implementation.

This project is not just a simple ML model—it is a high-performance, industry-standard solution that demonstrates expertise in data science, machine learning engineering, and financial analytics.

📜 Future Enhancements

🔹 Add LSTM (Long Short-Term Memory) model for time-series forecasting.
🔹 Incorporate sentiment analysis from financial news to improve predictions.
🔹 Deploy the model using a REST API for real-time predictions.
🔹 Optimize MLP architecture with additional regularization techniques.

💡 Final Thoughts

This project is a complete, end-to-end machine learning system that leverages both traditional regression techniques and deep learning. It showcases expertise in data preprocessing, feature engineering, model optimization, and performance analysis.

If you find this project useful, feel free to ⭐ Star the repository and contribute! 🚀

📩 Contact & Contributions

For any questions or improvements, feel free to reach out:

📧 erenaktuerk@hotmail.com
🌐 github.com/erenaktuerk

Want to contribute? Fork this repository, create a feature branch, and submit a pull request! 🚀
