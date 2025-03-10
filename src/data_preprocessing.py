import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataPreprocessing:
    """
    Handles data loading, cleaning, feature engineering, and saving processed data.
    """

    def __init__(self, input_file, output_file):
        """
        Initialize preprocessing with input and output file paths.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df = None  

    def load_data(self):
        """
        Loads data from a CSV file.
        """
        try:
            self.df = pd.read_csv(self.input_file)
            logging.info(f"✅ Data successfully loaded from {self.input_file}")
        except Exception as e:
            logging.error(f"❌ Error loading data: {e}")
            self.df = None

    def clean_data(self):
        """
        Cleans raw data: Converts date, selects relevant columns, removes invalid characters, 
        and ensures numeric types.
        """
        if self.df is None:
            logging.error("❌ No data loaded.")
            return

        # Convert 'Date' column to datetime and set as index
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df.set_index('Date', inplace=True)

        # Select only relevant columns
        self.df = self.df[['Close Price', 'Volume', 'Open Price', 'High', 'Low']]

        # Clean numeric columns (remove currency symbols, commas, and convert to float)
        self.df = self.df.replace({'\$': '', ',': ''}, regex=True).astype(float)

        # Drop rows with missing values
        self.df.dropna(inplace=True)

        logging.info("✅ Data cleaned successfully.")

    def feature_engineering(self):
        """
        Creates additional features including time-based attributes, moving averages, 
        volatility, and volume changes.
        """
        if self.df is None:
            logging.error("❌ No data available.")
            return

        # Time-based features
        self.df['Year'] = self.df.index.year
        self.df['Month'] = self.df.index.month
        self.df['Day'] = self.df.index.day
        self.df['DayOfWeek'] = self.df.index.dayofweek
        self.df['IsWeekend'] = (self.df['DayOfWeek'] >= 5).astype(int)

        # Moving averages & volatility
        self.df['SMA_7'] = self.df['Close Price'].rolling(window=7, min_periods=1).mean()
        self.df['EMA_7'] = self.df['Close Price'].ewm(span=7, adjust=False).mean()
        self.df['Volatility_7'] = self.df['Close Price'].rolling(window=7, min_periods=1).std()

        # Price changes & lagged values
        self.df['Daily_Return'] = self.df['Close Price'].pct_change()
        self.df['Close_Lag_1'] = self.df['Close Price'].shift(1)

        # Drop NaN values generated by feature creation
        self.df.dropna(inplace=True)

        logging.info("✅ Feature engineering completed.")

    def save_data(self):
        """
        Saves processed data to a CSV file.
        """
        try:
            self.df.to_csv(self.output_file)
            logging.info(f"✅ Processed data saved to {self.output_file}")
        except Exception as e:
            logging.error(f"❌ Error saving data: {e}")

    def preprocess_data(self):
        """
        Executes the full preprocessing pipeline.
        """
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.save_data()


# Run as standalone script
if __name__ == "__main__":
    logging.info("\n🚀 Starting Data Preprocessing...")
    processor = DataPreprocessing("data/raw/raw_share_data.csv", "data/processed/processed_share_data.csv")
    processor.preprocess_data()