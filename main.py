import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")  # Set the tracking URI
    print("Script execution started...")

    # Start MLflow run
    with mlflow.start_run():
        X_processed, y_processed, data_scaler, pca_processor = prepare_data()
        print("Data preparation finished")

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )

        # Log hyperparameters
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)

        try:
            mlflow.log_param("model_name", "Logistic Regression")  # Log the model name
            trained_log_reg_model = train_model(X_train, y_train)
            print("Model training finished")
        except Exception as e:
            mlflow.log_param("training_status", "failed")
            mlflow.log_metric("error", str(e))
            print(f"Model training failed: {e}")
            raise

        try:
            accuracy = evaluate_model(trained_log_reg_model, X_test, y_test)
            print("Model evaluation completed")
        except Exception as e:
            mlflow.log_param("evaluation_status", "failed")
            mlflow.log_metric("error", str(e))
            print(f"Model evaluation failed: {e}")
            raise

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        # Log additional metrics (precision, recall, F1-score)
        # ...

        # Log the model with input example
        input_example = X_test[0:1]  # Use the first sample as input example
        mlflow.sklearn.log_model(trained_log_reg_model, "model", input_example=input_example)

        save_model(trained_log_reg_model, data_scaler, pca_processor)

        final_model, final_scaler, final_pca = load_model()
        print("Script execution completed!")
