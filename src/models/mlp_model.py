import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class R2Score(tf.keras.metrics.Metric):
    """
    Custom R² (coefficient of determination) metric for regression evaluation.

    The R² score measures how well the model's predictions match the actual values.
    It is defined as:
    
        R² = 1 - (Sum of Squared Errors / Total Sum of Squares)
    
    where:
      - Sum of Squared Errors (SSE) = Σ(y_true - y_pred)²
      - Total Sum of Squares (SST) = Σ(y_true - mean(y_true))²
    
    A value close to 1 indicates a very good fit, while a negative value indicates that the model performs
    worse than simply predicting the mean.
    """
    def __init__(self, name='r2_score', **kwargs):
        super(R2Score, self).__init__(name=name, **kwargs)
        # Initialize accumulators for the sum of squared errors (numerator) and total sum of squares (denominator)
        self.squared_error = self.add_weight(name='squared_error', initializer='zeros')
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the metric state by computing and accumulating:
          - The sum of squared errors (SSE) for the current batch.
          - The total sum of squares (SST) for the current batch.
          
        Parameters:
            y_true (tensor): True target values.
            y_pred (tensor): Predicted values from the model.
        """
        squared_error = tf.reduce_sum(tf.square(y_true - y_pred))
        total_error = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        self.squared_error.assign_add(squared_error)
        self.total_error.assign_add(total_error)
    
    def result(self):
        """
        Computes the R² score based on the accumulated errors.
        
        Returns:
            A tensor representing the R² score.
        """
        return 1 - (self.squared_error / self.total_error)
    
    def reset_states(self):
        """
        Resets the metric's internal state variables for a new evaluation cycle.
        """
        self.squared_error.assign(0)
        self.total_error.assign(0)

class EpochLogger(tf.keras.callbacks.Callback):
    """
    Custom callback to log training progress every 10 epochs.
    
    This callback prints the training loss and validation loss every 10 epochs (or on the final epoch),
    providing periodic feedback during training while keeping the output uncluttered.
    """
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0 or (epoch + 1) == self.params.get('epochs'):
            loss = logs.get('loss')
            val_loss = logs.get('val_loss')
            print(f"Epoch {epoch + 1}: loss = {loss:.4f}, val_loss = {val_loss:.4f}")

class MLPModel:
    """
    Implements a Multi-Layer Perceptron (MLP) for regression using TensorFlow/Keras.

    This class supports:
      - Building a fully connected neural network.
      - Training with backpropagation, with training progress logged every 10 epochs.
      - Evaluating the model using key performance metrics.
      - Generating predictions on new data.
      - Extracting performance metrics in a standardized dictionary format.
    
    Model Architecture:
      - Input layer: 64 neurons with ReLU activation.
      - Hidden layer: 32 neurons with ReLU activation.
      - Output layer: 1 neuron for regression output.
    """
    def __init__(self, input_dim=None):
        """
        Initializes the MLP model.
        
        Parameters:
            input_dim (int, optional): The number of input features. This value is required to define the input layer.
        """
        self.model = None       # Will store the compiled Keras model.
        self.history = None     # Will hold the training history.
        self.X_train = None     # Will store training features for later reference.
        self.y_train = None     # Will store training labels.
        self.input_dim = input_dim  # Number of input features.

    def build(self):
        """
        Builds and compiles the MLP model using the Sequential API.

        Architecture:
            - A Dense layer with 64 neurons and ReLU activation (input layer).
            - A Dense layer with 32 neurons and ReLU activation (hidden layer).
            - A Dense layer with 1 neuron (output layer for regression).

        Compilation:
            - Optimizer: Adam.
            - Loss Function: Mean Squared Error (MSE).
            - Metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE),
              Root Mean Squared Error (RMSE), and a custom R² score.
        """
        self.model = Sequential([
            Dense(64, activation='relu', input_dim=self.input_dim),  # Input layer
            Dense(32, activation='relu'),                              # Hidden layer
            Dense(1)                                                   # Output layer
        ])
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=[
                MeanAbsoluteError(),
                MeanSquaredError(),
                RootMeanSquaredError(name="root_mean_squared_error"),
                R2Score()
            ]
        )
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Trains the MLP model on the provided training data.
        
        Parameters:
            X_train (array-like): The training feature set.
            y_train (array-like): The training labels.
            epochs (int, optional): Number of training epochs (default: 50).
            batch_size (int, optional): Number of samples per training step (default: 32).
        
        The training data is automatically split into 80% training and 20% validation.
        A custom callback (EpochLogger) logs the loss and validation loss every 10 epochs.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.build()  # Build and compile the model
        # Instantiate the custom callback to print every 10 epochs.
        epoch_logger = EpochLogger()
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,  # Suppress default output
            callbacks=[epoch_logger]
        )
    
    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained MLP model on the test dataset.
        
        Parameters:
            X_test (array-like): Test feature set.
            y_test (array-like): True labels for the test set.
        
        Returns:
            A list of evaluation metrics (loss, MAE, MSE, RMSE, R²) from model.evaluate().
        """
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X_input):
        """
        Generates predictions for the given input data.
        
        Parameters:
            X_input (array-like): Input features for prediction.
        
        Returns:
            A NumPy array containing the predicted values.
        """
        return self.model.predict(X_input, verbose=0)
    
    def get_metrics(self, X_train, y_train, X_test, y_test):
        """
        Computes performance metrics for both the training and test datasets.
        
        The returned dictionary uses hyphen-separated keys to conform to the expected format.
        
        Metrics computed:
            - "train-mse": Mean Squared Error on the training set.
            - "test-mse": Mean Squared Error on the test set.
            - "train-rmse": Root Mean Squared Error on the training set.
            - "test-rmse": Root Mean Squared Error on the test set.
            - "train-mae": Mean Absolute Error on the training set.
            - "test-mae": Mean Absolute Error on the test set.
            - "train-r2": R² score on the training set.
            - "test-r2": R² score on the test set.
        
        Parameters:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
        
        Returns:
            dict: A dictionary containing the computed metrics.
        """
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        metrics = {
            "train-mse": mean_squared_error(y_train, y_train_pred),
            "test-mse": mean_squared_error(y_test, y_test_pred),
            "train-rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "test-rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "train-mae": mean_absolute_error(y_train, y_train_pred),
            "test-mae": mean_absolute_error(y_test, y_test_pred),
            "train-r2": r2_score(y_train, y_train_pred),
            "test-r2": r2_score(y_test, y_test_pred)
        }
        return metrics