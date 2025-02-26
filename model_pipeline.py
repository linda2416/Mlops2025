import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import logging
import joblib
from mlflow.models.signature import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

MLFLOW_EXPERIMENT = "Churn Prediction Logistic"


def setup_mlflow():
    """Configure MLflow tracking avec gestion des runs actives"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    # Do not end the main run here to allow nested runs to function correctly


def prepare_data():
    """Charge et prépare les données avec tracking MLflow"""
    logging.info("Starting data preparation...")
    setup_mlflow()
    try:
        with mlflow.start_run(nested=True, run_name="Data Preparation") as run:
            logging.info(f"Nested run ID: {run.info.run_id}")
            logging.info("Logging and loading data...")
            mlflow.log_artifact("merged_churn.csv")
            df = pd.read_csv("merged_churn.csv")

            # Log des métadonnées initiales
            mlflow.log_params(
                {
                    "dataset_rows": df.shape[0],
                    "initial_features": df.shape[1],
                    "missing_values": df.isnull().sum().sum(),
                    "duplicates": df.duplicated().sum(),
                }
            )

            # Préprocessing
            target = df["Churn"]
            features = df.drop(columns=["Churn"])

            # Encodage
            features["State"] = features["State"].map(
                features["State"].value_counts().to_dict()
            )
            features["International plan"] = features["International plan"].map(
                {"Yes": 1, "No": 0}
            )
            features["Voice mail plan"] = features["Voice mail plan"].map(
                {"Yes": 1, "No": 0}
            )

            # Standardisation
            feature_scaler = StandardScaler()
            features_scaled = feature_scaler.fit_transform(features)

            # Log des paramètres de preprocessing
            mlflow.log_params(
                {
                    "scaler_type": type(feature_scaler).__name__,
                    "scaled_features": features_scaled.shape[1],
                }
            )

            logging.info("Data preparation completed successfully.")
            return features_scaled, target, feature_scaler, None

    except Exception as e:
        mlflow.end_run(status="FAILED")
        logging.error(f"Échec préparation données : {str(e)}")
        raise


def train_model(train_features, train_labels):
    """Entraîne le modèle avec tracking MLflow complet"""
    setup_mlflow()
    try:
        with mlflow.start_run(nested=True, run_name="Model Training"):
            # PCA
            pca_transformer = PCA(n_components=0.95)
            train_features_pca = pca_transformer.fit_transform(train_features)

            mlflow.log_params(
                {
                    "pca_n_components": 0.95,
                    "pca_explained_variance": round(
                        np.array(pca_transformer.explained_variance_ratio_).sum(), 3
                    ),
                    "pca_components": train_features_pca.shape[1],
                }
            )

            # Rééquilibrage SMOTE + ENN
            smote = SMOTE(random_state=42)
            X_resampled, Y_resampled = smote.fit_resample(
                train_features_pca, train_labels
            )
            enn = EditedNearestNeighbours()
            X_resampled_enn, Y_resampled_enn = enn.fit_resample(
                X_resampled, Y_resampled
            )

            mlflow.log_params(
                {
                    "sampling_strategy": "SMOTE+ENN",
                    "resampled_samples": X_resampled_enn.shape[0],
                    "smote_random_state": 42,
                }
            )

            # Entraînement modèle
            log_reg_model = LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced", solver="lbfgs"
            )

            log_reg_model.fit(X_resampled_enn, Y_resampled_enn)

            # Log des hyperparamètres
            mlflow.log_params(
                {
                    "model_type": "LogisticRegression",
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "class_weight": "balanced",
                }
            )

            # Signature du modèle
            input_example = X_resampled_enn[:1]
            signature = infer_signature(
                input_example, log_reg_model.predict(input_example)
            )

            # Enregistrement MLflow
            mlflow.sklearn.log_model(
                sk_model=log_reg_model,
                artifact_path="churn_model",
                signature=signature,
                registered_model_name="ChurnPredictionLogistic",
                input_example=input_example,
            )

            return log_reg_model, pca_transformer

    except Exception as e:
        mlflow.end_run(status="FAILED")
        logging.error(f"Échec entraînement modèle : {str(e)}")
        raise


def evaluate_model(model_instance, test_features, test_labels):
    """Évaluation complète avec tracking MLflow"""
    setup_mlflow()
    try:
        with mlflow.start_run(nested=True, run_name="Model Evaluation"):
            # Prédictions
            predictions = model_instance.predict(test_features)

            # Calcul des métriques
            accuracy = accuracy_score(test_labels, predictions)
            report = classification_report(test_labels, predictions, output_dict=True)
            conf_matrix = confusion_matrix(test_labels, predictions)

            # Log des métriques et des prédictions
            logging.info(f"Predictions: {predictions}")
            logging.info(f"Unique classes in predictions: {np.unique(predictions)}")
            precision_0 = report["0"]["precision"] if "0" in report else 0
            recall_0 = report["0"]["recall"] if "0" in report else 0
            f1_0 = report["0"]["f1-score"] if "0" in report else 0
            precision_1 = report["1"]["precision"] if "1" in report else 0
            recall_1 = report["1"]["recall"] if "1" in report else 0
            f1_1 = report["1"]["f1-score"] if "1" in report else 0

            mlflow.log_metrics(
                {
                    "accuracy": accuracy,
                    "precision_0": precision_0,
                    "recall_0": recall_0,
                    "f1_0": f1_0,
                    "precision_1": precision_1,
                    "recall_1": recall_1,
                    "f1_1": f1_1,
                }
            )

            # Matrice de confusion
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Non Churn", "Churn"],
                yticklabels=["Non Churn", "Churn"],
            )
            plt.xlabel("Prédit")
            plt.ylabel("Réel")
            plt.title("Matrice de Confusion")
            plot_path = "confusion_matrix.png"
            plt.savefig(plot_path)
            plt.close()

            mlflow.log_artifact(plot_path)

            return accuracy

    except Exception as e:
        mlflow.end_run(status="FAILED")
        logging.error(f"Échec évaluation modèle : {str(e)}")
        raise


def save_model(model_instance, feature_scaler, pca_transformer):
    """Sauvegarde des artefacts locaux avec logging"""
    try:
        joblib.dump(model_instance, "model.pkl")
        joblib.dump(feature_scaler, "scaler.pkl")
        if pca_transformer:
            joblib.dump(pca_transformer, "pca.pkl")
        mlflow.log_artifacts(".", artifact_path="local_artifacts")
    except Exception as e:
        logging.error(f"Échec sauvegarde modèle : {str(e)}")
        raise


def load_model():
    """Chargement des artefacts locaux"""
    try:
        return (
            joblib.load("model.pkl"),
            joblib.load("scaler.pkl"),
            joblib.load("pca.pkl"),
        )
    except Exception as e:
        logging.error(f"Échec chargement modèle : {str(e)}")
        raise


if __name__ == "__main__":
    print("Script execution started...")

    X_processed, y_processed, data_scaler, pca_processor = prepare_data()
    print("Data preparation finished")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42
    )

    trained_log_reg_model = train_model(X_train, y_train)
    print("Model training finished")

    evaluate_model(trained_log_reg_model, X_test, y_test)
    print("Model evaluation completed")

    save_model(trained_log_reg_model, data_scaler, pca_processor)

    final_model, final_scaler, final_pca = load_model()
    print("Script execution completed!")
