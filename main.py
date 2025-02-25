import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="ML Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--save", action="store_true", help="Save model")
    parser.add_argument("--load", action="store_true", help="Load model")
    return parser.parse_args()


def run_pipeline(args, tracking_uri="http://localhost:5001"):
    mlflow.set_tracking_uri(tracking_uri)
    print("Script execution started... Ensure to push changes to GitHub for version control.")

    with mlflow.start_run():
        if args.prepare:
            X_processed, y_processed, data_scaler, pca_processor = prepare_data()
            print("Data preparation finished")

        if args.train:
            # Ensure data is prepared only if training
            if "X_processed" not in locals():
                X_processed, y_processed, data_scaler, pca_processor = prepare_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_iter", 1000)

        if args.train:
            # Ensure data is prepared only if training
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_iter", 1000)
            try:
                mlflow.log_param("model_name", "Logistic Regression")
                trained_log_reg_model = train_model(X_train, y_train)
                print("Model training finished")
                return trained_log_reg_model, X_test, y_test
            except Exception as e:
                mlflow.log_param("training_status", "failed")
                mlflow.log_metric("error", str(e))
                print(f"Model training failed: {e}")
                raise

        if args.evaluate:
            # End any active run before starting evaluation
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run():
                # Load trained model and test data
                loaded_model, loaded_scaler, loaded_pca = load_model()
                X_processed, y_processed, _, _ = prepare_data()
                X_test = loaded_pca.transform(loaded_scaler.transform(X_processed))
                try:
                    accuracy = evaluate_model(loaded_model, X_test, y_processed)
                    print("Model evaluation completed")
                    mlflow.log_metric("accuracy", accuracy)
                except Exception as e:
                    mlflow.log_param("evaluation_status", "failed")
                    mlflow.log_metric("error", str(e))
                    print(f"Model evaluation failed: {e}")
                    raise

        if args.save:
            trained_log_reg_model, _, _ = run_pipeline(argparse.Namespace(train=True))
            X_processed, _, data_scaler, pca_processor = prepare_data()
            save_model(trained_log_reg_model, data_scaler, pca_processor)

        if args.load:
            final_model, final_scaler, final_pca = load_model()
            print("Model loaded successfully!")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)
    print("Script execution completed!")
