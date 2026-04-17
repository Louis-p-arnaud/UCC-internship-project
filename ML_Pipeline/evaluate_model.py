import os
import json
import pickle
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from path import DATA_PATH

# 1. Configuration des chemins
EMBED_DIR = os.path.join(DATA_PATH,"wearable_downstream/wesad/wesad_stress_wav_embed")
MODEL_PATH = 'saved_models/fatigue_RandomForest.joblib'
SPLIT_PATH = os.path.join(DATA_PATH, "wearable_downstream", "wesad", "train_test_split.json")


def load_test_data(file_id_list):
    """Charge uniquement les données nécessaires pour l'évaluation."""
    X, y = [], []
    for file_id in file_id_list:
        file_path = os.path.join(EMBED_DIR, file_id)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                feat = data['embed'].astype(np.float32).flatten()
                if feat.shape[0] == 7680:
                    X.append(feat)
                    y.append(data['label'][0]['class'])
    return np.array(X), np.array(y)


def run_evaluation(MODEL_PATH=MODEL_PATH):
    # A. Charger le modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Le modèle {MODEL_PATH} est introuvable. Lance d'abord l'entraînement.")
        return

    print("Chargement du modèle...")
    clf = joblib.load(MODEL_PATH)

    # B. Charger les données de test
    print("Chargement des données de test...")
    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)
    X_test, y_test = load_test_data(split['test'])

    # C. Prédiction
    y_pred = clf.predict(X_test)

    # D. Métriques
    target_names = ['Baseline', 'Stress', 'Amusement']
    print("\n" + "=" * 30)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 30)
    print(f"F1-Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # E. Matrice de confusion
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion matrix (Random Forest)')
    plt.ylabel('Reality')
    plt.xlabel('Prediction')
    plt.show()


if __name__ == "__main__":
    run_evaluation(MODEL_PATH=MODEL_PATH)