# def test_dummy(): assert 2 + 2 == 4

# tests/test_model_pipeline.py
import pytest
from unittest.mock import MagicMock
from model_pipeline import train_model
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.linear_model import LogisticRegression


# Test unitaire pour la fonction train_model
def test_train_model(mocker):
    # Créer des données factices pour l'entraînement
    X_train, y_train = make_classification(
        n_samples=100, n_features=20, random_state=42
    )

    # Mocking des dépendances externes
    mock_pca = mocker.patch("model_pipeline.PCA", return_value=MagicMock(spec=PCA))
    mock_pca.return_value.explained_variance_ratio_ = [0.95]  # Set the explained_variance_ratio_
    mock_pca.return_value.n_components = 0.95  # Set the n_components attribute
    mock_smote = mocker.patch(
        "model_pipeline.SMOTE", return_value=MagicMock(spec=SMOTE)
    )
    mock_smote.return_value.fit_resample.return_value = (X_train, y_train)  # Ensure it returns the expected values
    mock_enn = mocker.patch(
        "model_pipeline.EditedNearestNeighbours",
        return_value=MagicMock(spec=EditedNearestNeighbours),
    )
    mock_log_reg = mocker.patch(
        "model_pipeline.LogisticRegression",
        return_value=MagicMock(spec=LogisticRegression),
    )

    # Simuler le comportement de PCA (par exemple, on suppose qu'il transforme les données)
    mock_pca.return_value.fit_transform.return_value = X_train  # X_train après PCA

    # Simuler le comportement de SMOTE et ENN
    mock_smote.return_value.fit_resample.return_value = (X_train, y_train)
    mock_enn.return_value.fit_resample.return_value = (X_train, y_train)

    # Appeler la fonction train_model
    model, pca_transformer = train_model(X_train, y_train)

    # Vérifications :
    # Assurez-vous que PCA, SMOTE et ENN ont été appelés
    mock_pca.return_value.fit_transform.assert_called_once_with(X_train)
    mock_smote.return_value.fit_resample.assert_called_once_with(X_train, y_train)
    mock_enn.return_value.fit_resample.assert_called_once_with(X_train, y_train)

    # Vérification que le modèle est bien un LogisticRegression
    assert isinstance(model, LogisticRegression)
    # Vérification que le PCA a le bon nombre de composantes
    assert pca_transformer.n_components == 0.95
    # Vérification que le modèle a bien été ajusté (fit)
    mock_log_reg.return_value.fit.assert_called_once_with(X_train, y_train)
