import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yaml
from loguru import logger
import mlflow
import mlflow.sklearn
from src.model.train_model import load_data, extract_features

def tune_model(df, features, target):
    """Performs hyperparameter tuning and logs experiments with MLflow."""
    logger.info("Starting hyperparameter tuning with MLflow tracking")

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.log_params(param_grid)

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Log the best parameters and score
        mlflow.log_param("best_params", grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric("mse", mse)

        # Log the model
        mlflow.sklearn.log_model(best_model, "random_forest_model")

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"MSE on test set: {mse}")

    return best_model

def main():
    """Main function to run the hyperparameter tuning."""
    with open("settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    PROCESSED_DATA_PATH = config['PROCESSED_DATA_PATH']

    df = load_data(PROCESSED_DATA_PATH)
    df = extract_features(df)

    features = ['feature1', 'feature2']
    target = 'target'

    tune_model(df, features, target)

if __name__ == "__main__":
    main()
