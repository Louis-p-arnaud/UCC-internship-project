import os
import json
import pickle
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Classifieurs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier,
                              BaggingClassifier, HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from path import DATA_PATH

# On ignore les warnings de convergence pour ne pas polluer la console pendant ton repas
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
EMBED_DIR = os.path.join(DATA_PATH, "wearable_downstream/wesad/wesad_stress_wav_embed")
SPLIT_PATH = os.path.join(DATA_PATH, "wearable_downstream", "wesad", "train_test_split.json")
SAVE_DIR = "benchmark_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. LISTE DES 30+ MODÈLES/CONFIGURATIONS ---
models_to_test = [
    # Arbres & Forêts
    ("DecisionTree_Gini", DecisionTreeClassifier(criterion='gini', random_state=42)),
    ("DecisionTree_Entropy", DecisionTreeClassifier(criterion='entropy', random_state=42)),
    ("RandomForest_100", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ("RandomForest_300", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)),
    ("ExtraTrees_100", ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=42)),

    # Boosting
    ("AdaBoost_Base", AdaBoostClassifier(n_estimators=50, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("HistGradientBoosting", HistGradientBoostingClassifier(max_iter=100, random_state=42)),

    # Modèles Linéaires (Nécessitent StandardScaler)
    ("LogisticRegression_L2", LogisticRegression(max_iter=1000, penalty='l2')),
    ("RidgeClassifier", RidgeClassifier()),
    ("SGD_Classifier", SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)),
    ("LinearSVC", LinearSVC(max_iter=1000, random_state=42)),

    # SVM (Plus lents sur 7680 features)
    ("SVM_RBF", SVC(kernel='rbf', C=1.0, random_state=42)),
    ("SVM_Poly", SVC(kernel='poly', degree=3, random_state=42)),

    # Voisins & Bayésiens
    ("KNN_3_Euclidean", KNeighborsClassifier(n_neighbors=3)),
    ("KNN_5_Euclidean", KNeighborsClassifier(n_neighbors=5)),
    ("KNN_5_Distance", KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ("GaussianNB", GaussianNB()),

    # Analyses Discriminantes
    ("LDA", LinearDiscriminantAnalysis()),
    # QDA est souvent instable avec beaucoup de features, on verra s'il passe
    ("QDA", QuadraticDiscriminantAnalysis()),

    # Réseaux de neurones (MLP)
    ("MLP_S1", MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, max_iter=500, random_state=42)),
    ("MLP_S2", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)),
    ("MLP_S3", MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)),
    ("MLP_Tanh", MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=500, random_state=42)),

    # Bagging
    ("Bagging_DT", BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)),

    # Variantes de régularisation et hyperparamètres
    ("RandomForest_Balanced", RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)),
    ("SVM_Weighted", SVC(kernel='rbf', class_weight='balanced', random_state=42)),
    ("Logistic_Balanced", LogisticRegression(max_iter=1000, class_weight='balanced')),
    ("DecisionTree_Depth10", DecisionTreeClassifier(max_depth=10, random_state=42)),
    ("MLP_StrongReg", MLPClassifier(hidden_layer_sizes=(100,), alpha=0.1, max_iter=500, random_state=42))
]


def load_data(file_id_list):
    X, y = [], []
    for file_id in file_id_list:
        file_path = os.path.join(EMBED_DIR, file_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                feat = data['embed'].astype(np.float32).flatten()

                # Nettoyage NaN / Inf
                if np.isnan(feat).any() or np.isinf(feat).any():
                    continue

                if feat.shape[0] == 7680:
                    X.append(feat)
                    y.append(data['label'][0]['class'])
            except Exception:
                continue
    return np.array(X), np.array(y)


def run_benchmark():
    # 1. Chargement et Scaling
    print("--- Chargement des données ---")
    with open(SPLIT_PATH, 'r') as f:
        split = json.load(f)
    X_train, y_train = load_data(split['train'])
    X_test, y_test = load_data(split['test'])

    print(f"Dataset chargé: Train={len(X_train)}, Test={len(X_test)}")

    print("--- Scaling des données (StandardScaler) ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    target_names = ['Baseline', 'Stress', 'Amusement']
    results = []

    # 2. Boucle de Benchmark
    for name, clf in models_to_test:
        print(f"\n[TESTING] {name}...", end=" ", flush=True)
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            score = f1_score(y_test, y_pred, average='weighted')
            results.append({'name': name, 'score': score})
            print(f"DONE (F1: {score:.4f})")

            # Matrice de confusion
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='blues',
                        xticklabels=target_names, yticklabels=target_names)
            plt.title(f'CM: {name} (F1: {score:.4f})')
            plt.savefig(os.path.join(SAVE_DIR, f"cm_{name}.png"))
            plt.close()

        except Exception as e:
            print(f"FAILED | Erreur: {e}")

    # 3. Résumé Final
    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "=" * 50)
    print("RANKING FINAL DES MODÈLES")
    print("=" * 50)
    for i, res in enumerate(results[:10]):  # Top 10
        print(f"{i + 1}. {res['name']}: {res['score']:.4f}")

    if results:
        print("\n🏆 LE GAGNANT :", results[0]['name'], "avec", round(results[0]['score'], 4))

    # Sauvegarde du tableau des scores en JSON pour analyse plus tard
    with open(os.path.join(SAVE_DIR, "results_summary.json"), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    run_benchmark()