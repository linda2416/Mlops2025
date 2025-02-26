import argparse
import mlflow
from sklearn.model_selection import train_test_split
from model_pipeline import (
    setup_mlflow,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def execute_pipeline(args):
    """Exécute le pipeline complet avec gestion des runs MLflow"""
    setup_mlflow()

    with mlflow.start_run(run_name="Main Pipeline"):
        # Préparation des données
        if args.prepare:
            with mlflow.start_run(nested=True, run_name="Data Preparation"):
                prepare_data()

        # Entraînement
        if args.train:
            with mlflow.start_run(nested=True, run_name="Model Training"):
                features, target, scaler, _ = prepare_data()
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                mlflow.log_param("test_size", 0.2)
                model, pca = train_model(X_train, y_train)

        # Évaluation
        if args.evaluate:
            with mlflow.start_run(nested=True, run_name="Model Evaluation"):
                model, scaler, pca = load_model()
                features, target, _, _ = prepare_data()
                features_pca = pca.transform(scaler.transform(features))
                accuracy = evaluate_model(model, features_pca, target)
                mlflow.log_metric("final_accuracy", accuracy)

        # Sauvegarde
        if args.save:
            with mlflow.start_run(nested=True, run_name="Model Saving"):
                model, scaler, pca = load_model()
                save_model(model, scaler, pca)
                mlflow.log_artifact("model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de prédiction de churn")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle")

    args = parser.parse_args()

    try:
        execute_pipeline(args)
        print("✅ Pipeline exécuté avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {str(e)}")
        raise
