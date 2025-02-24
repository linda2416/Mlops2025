import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import mlflow
import logging

def prepare_data():
    """Load and preprocess the dataset."""
    print("Starting data preparation...")

    df = pd.read_csv("merged_churn.csv")
    print("Dataset loaded successfully!")

    print("Missing values per column:\n", df.isnull().sum())
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    target = df["Churn"]
    features = df.drop(columns=["Churn"])
    print("Target variable separated")

    print("Applying frequency encoding for 'State'...")
    features["State"] = features["State"].map(features["State"].value_counts().to_dict())
    print("Frequency encoding applied")

    print("Converting categorical variables to numeric...")
    features["International plan"] = features["International plan"].map({"Yes": 1, "No": 0})
    features["Voice mail plan"] = features["Voice mail plan"].map({"Yes": 1, "No": 0})
    print("Categorical conversion completed")

    print("Applying standardization...")
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    print("Standardization completed")

    print("Data preprocessing completed!")
    return features_scaled, target, feature_scaler, None

def train_model(train_features, train_labels):
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting model training...")
    """Train a Logistic Regression model with SMOTE and ENN for handling class imbalance."""
    try:
        # Apply PCA transformation
        pca_transformer = PCA(n_components=0.95)
        train_features = pca_transformer.fit_transform(train_features)
        
        # Log PCA parameters
        mlflow.log_param("pca_n_components", 0.95)
        mlflow.log_param("pca_explained_variance", sum(pca_transformer.explained_variance_ratio_))
        
        print("Shape of data before SMOTE:", train_features.shape)
        print("Shape of labels before SMOTE:", train_labels.shape)

        # Apply SMOTE for class balancing
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, Y_resampled = smote.fit_resample(train_features, train_labels)
        print("Shape after SMOTE:", X_resampled.shape)

        # Apply ENN for further class balancing
        print("Applying ENN for further class balancing...")
        enn = EditedNearestNeighbours()
        X_resampled_enn, Y_resampled_enn = enn.fit_resample(X_resampled, Y_resampled)
        print(f"SMOTE + ENN applied: {X_resampled_enn.shape[0]} samples after resampling")

        # Train the Logistic Regression model
        logging.info("Training the Logistic Regression model...")
        log_reg_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        
        # Log model parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        
        log_reg_model.fit(X_resampled_enn, Y_resampled_enn)
        logging.info("Model training completed!")

        return log_reg_model, pca_transformer

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

def evaluate_model(model_instance, test_features, test_labels):
    """Evaluate the model's performance."""
    print("Evaluating the model...")

    # Generate predictions
    predictions = model_instance.predict(test_features)

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(test_labels, predictions)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_false", report['False']['precision'])
    mlflow.log_metric("precision_true", report['True']['precision'])
    mlflow.log_metric("recall_false", report['False']['recall'])
    mlflow.log_metric("recall_true", report['True']['recall'])
    mlflow.log_metric("f1_false", report['False']['f1-score'])
    mlflow.log_metric("f1_true", report['True']['f1-score'])
    
    # Log confusion matrix
    mlflow.log_artifact(plot_confusion_matrix(conf_matrix), "confusion_matrix.png")

    # Print out the results
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(test_labels, predictions))
    print("Confusion Matrix:\n", conf_matrix)

    # Return accuracy for logging
    return accuracy

def plot_confusion_matrix(conf_matrix):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    file_path = 'confusion_matrix.png'
    plt.savefig(file_path)
    plt.close()
    return file_path

def save_model(model_instance, feature_scaler, pca_transformer):
    """Save model and preprocessing artifacts using pickle."""
    print("Saving model and preprocessing artifacts...")
    with open("./model.pkl", "wb") as model_file:
        pickle.dump(model_instance, model_file)
    with open("./scaler.pkl", "wb") as scaler_file:
        pickle.dump(feature_scaler, scaler_file)
    if pca_transformer:
        with open("./pca.pkl", "wb") as pca_file:
            pickle.dump(pca_transformer, pca_file)
    print("Model, scaler, and PCA saved successfully!")

def load_model():
    """Load the trained model, scaler, and PCA."""
    print("Loading model and preprocessing artifacts...")
    with open("./model.pkl", "rb") as model_file:
        loaded_model_instance = pickle.load(model_file)
    with open("./scaler.pkl", "rb") as scaler_file:
        loaded_scaler_instance = pickle.load(scaler_file)
    with open("./pca.pkl", "rb") as pca_file:
        loaded_pca_instance = pickle.load(pca_file)
    print("Model, scaler, and PCA loaded successfully!")
    return loaded_model_instance, loaded_scaler_instance, loaded_pca_instance

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
