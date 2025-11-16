import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import dump
import yaml
from loguru import logger
import re
import numpy as np

def load_data(file_path):
    """Loads the processed OCR data."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def extract_features(df):
    """Extracts features and a target from the OCR text."""
    logger.info("Extracting features from OCR text")

    # Placeholder feature extraction: find all numbers in the text
    def find_numbers(text):
        if isinstance(text, str):
            # Find all numbers (including decimals)
            numbers = re.findall(r'[+-]?\d+\.?\d*', text)
            return [float(n) for n in numbers]
        return []

    df['numbers'] = df['ocr_text'].apply(find_numbers)

    # Create features: for simplicity, use first two numbers found as features
    # and the third as the target. This is a placeholder and should be replaced
    # with a more robust feature engineering process.
    df['feature1'] = df['numbers'].apply(lambda x: x[0] if len(x) > 0 else 0)
    df['feature2'] = df['numbers'].apply(lambda x: x[1] if len(x) > 1 else 0)
    df['target'] = df['numbers'].apply(lambda x: x[2] if len(x) > 2 else 0)

    # Create a categorical feature for bias detection slicing
    df['category'] = np.random.choice(['A', 'B', 'C'], size=len(df))

    return df

def train_model(df, features, target):
    """Trains a simple linear regression model."""
    logger.info("Training the model")
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def validate_model(model, X_test, y_test):
    """Validates the model and returns performance metrics."""
    logger.info("Validating the model")
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    logger.info(f"Model R2 Score: {r2}")
    return r2

def detect_bias(model, df, features, target, slice_column='category'):
    """Performs a simple bias check by slicing the data."""
    logger.info(f"Performing bias detection on column: {slice_column}")

    results = {}
    for slice_value in df[slice_column].unique():
        sliced_df = df[df[slice_column] == slice_value]
        X_slice = sliced_df[features]
        y_slice = sliced_df[target]

        if len(sliced_df) > 10: # Only evaluate if there's enough data
            predictions = model.predict(X_slice)
            r2 = r2_score(y_slice, predictions)
            results[slice_value] = r2
            logger.info(f"R2 score for slice '{slice_value}': {r2}")
        else:
            logger.warning(f"Skipping slice '{slice_value}' due to insufficient data.")

    return results

def save_model(model, path):
    """Saves the trained model to a file."""
    logger.info(f"Saving model to {path}")
    dump(model, path)

def main():
    """Main function to run the model training pipeline."""
    with open("settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    PROCESSED_DATA_PATH = config['PROCESSED_DATA_PATH']
    MODEL_PATH = config['MODEL_PATH']
    MODEL_NAME = config['MODEL_NAME']

    df = load_data(PROCESSED_DATA_PATH)
    df = extract_features(df)

    features = ['feature1', 'feature2']
    target = 'target'

    model, X_test, y_test = train_model(df, features, target)
    validate_model(model, X_test, y_test)
    detect_bias(model, df, features, target)

    model_filepath = f"{MODEL_PATH}/{MODEL_NAME}"
    save_model(model, model_filepath)

if __name__ == "__main__":
    main()
