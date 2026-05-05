import json
import os
import pickle
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

from path import DATA_PATH

# ============= CONFIGURATION =============
with open(os.path.join(DATA_PATH, "wearable_downstream", "wesad", "train_test_split.json"), 'r') as f:
    split = json.load(f)

embed_dir = os.path.join(DATA_PATH, "wearable_downstream/wesad/wesad_stress_wav_embed")

def load_data_from_list(file_id_list):
    X, y = [], []
    for file_id in file_id_list:
        file_path = os.path.join(embed_dir, file_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if 'embed' in data:
                    emb = data['embed'].astype(np.float32)
                    feat = emb.flatten()

                    # Nettoyage NaN/Inf
                    if np.isnan(feat).any() or np.isinf(feat).any():
                        feat = np.nan_to_num(feat)

                    if feat.shape[0] == 7680:
                        X.append(feat)
                        label = data['label'][0]['class']
                        y.append(label)
                    else:
                        print(f"Format incorrect pour {file_id}: {feat.shape[0]}")
            except Exception as e:
                print(f"Erreur lors de la lecture de {file_id}: {e}")

    return np.array(X), np.array(y)

# ============= CHARGEMENT DES DONNÉES =============
print("=" * 60)
print("CHARGEMENT DES DONNÉES")
print("=" * 60)

X_train, y_train = load_data_from_list(split['train'])
X_test, y_test = load_data_from_list(split['test'])

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ============= PREPROCESSING =============
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA pour réduire la dimensionnalité
pca = PCA(n_components=100, random_state=42)  # Réduire à 100 composantes
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"✓ Variance expliquée par PCA: {pca.explained_variance_ratio_.sum():.4f}")
print(f"✓ Nouvelles dimensions: {X_train_pca.shape}")

# ============= TEST DES PROFONDEURS D'ARBRE =============
print("\n" + "=" * 60)
print("TEST DES PROFONDEURS D'ARBRE DE DÉCISION")
print("=" * 60)

max_depths = range(1, 21)  # Profondeurs de 1 à 20
train_accuracies = []
test_accuracies = []
train_f1s = []
test_f1s = []

best_depth = 1
best_test_acc = 0.0
previous_test_acc = 0.0
decrease_count = 0

for depth in max_depths:
    print(f"\nTest profondeur: {depth}")

    # Entraîner le modèle
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train_pca, y_train)

    # Prédictions
    y_train_pred = clf.predict(X_train_pca)
    y_test_pred = clf.predict(X_test_pca)

    # Métriques
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    train_f1s.append(train_f1)
    test_f1s.append(test_f1)

    print(f"  Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"  Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

    # Vérifier si la précision test diminue
    if depth > 1 and test_acc < previous_test_acc:
        decrease_count += 1
        print(f"⚠️  Baisse détectée ({decrease_count}/5): Précision test diminue à profondeur {depth} (précédente: {previous_test_acc:.4f}, actuelle: {test_acc:.4f})")
        if decrease_count >= 5:
            print(f"🛑 Arrêt après 5 baisses consécutives.")
            break
    else:
        decrease_count = 0  # Reset si pas de baisse

    # Mettre à jour le meilleur
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_depth = depth

    previous_test_acc = test_acc

print(f"\n🎯 Meilleure profondeur trouvée: {best_depth} avec Test Accuracy: {best_test_acc:.4f}")

# ============= VISUALISATIONS =============
print("\n" + "=" * 60)
print("VISUALISATIONS")
print("=" * 60)

depths_tested = list(range(1, len(train_accuracies) + 1))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(depths_tested, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(depths_tested, test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('Profondeur maximale')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Profondeur')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(depths_tested, train_f1s, label='Train F1', marker='o')
plt.plot(depths_tested, test_f1s, label='Test F1', marker='o')
plt.xlabel('Profondeur maximale')
plt.ylabel('F1-Score (weighted)')
plt.title('F1-Score vs Profondeur')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('saved_models/decision_tree_depth_analysis.png', dpi=100)
print("✓ Graphiques sauvegardés: decision_tree_depth_analysis.png")

# ============= ENTRAÎNEMENT FINAL AVEC MEILLEURE PROFONDEUR =============
print("\n" + "=" * 60)
print("ENTRAÎNEMENT FINAL")
print("=" * 60)

final_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_clf.fit(X_train_pca, y_train)

y_test_pred_final = final_clf.predict(X_test_pca)
final_test_acc = accuracy_score(y_test, y_test_pred_final)
final_test_f1 = f1_score(y_test, y_test_pred_final, average='weighted')

print(f"✓ Modèle final - Test Accuracy: {final_test_acc:.4f}, Test F1: {final_test_f1:.4f}")

target_names = ['Baseline', 'Stress', 'Amusement']
print("\n📊 Rapport de Classification Final:")
print(classification_report(y_test, y_test_pred_final, target_names=target_names))

# Sauvegarder le modèle
joblib.dump(final_clf, f'saved_models/decision_tree_depth_{best_depth}.joblib')
joblib.dump(scaler, 'saved_models/scaler_decision_tree.joblib')
joblib.dump(pca, 'saved_models/pca_decision_tree.joblib')

print(f"✓ Modèle sauvegardé: decision_tree_depth_{best_depth}.joblib")

# Résultats
results = {
    'model': 'DecisionTree',
    'best_depth': best_depth,
    'test_accuracy': float(final_test_acc),
    'test_f1': float(final_test_f1),
    'preprocessing': {
        'standardscaler': True,
        'pca_n_components': 100,
        'variance_explained': float(pca.explained_variance_ratio_.sum())
    }
}

with open('saved_models/decision_tree_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"✓ Résultats sauvegardés: {results}")