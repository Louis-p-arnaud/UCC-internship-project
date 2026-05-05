import json
import os
import pickle
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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


# ============= 1. CHARGEMENT DES DONNÉES =============
print("=" * 60)
print("1️⃣  CHARGEMENT DES DONNÉES")
print("=" * 60)

X_train, y_train = load_data_from_list(split['train'])
X_test, y_test = load_data_from_list(split['test'])

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Vérifier la distribution des classes
train_dist = np.bincount(y_train) / len(y_train)
test_dist = np.bincount(y_test) / len(y_test)
print(f"✓ Distribution train: {train_dist}")
print(f"✓ Distribution test: {test_dist}")

# ============= 2. PREPROCESSING =============
print("\n" + "=" * 60)
print("2️⃣  PREPROCESSING")
print("=" * 60)

# Standardisation
print("  → StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Réduction de dimensionnalité (PCA)
print("  → PCA (7680 → 256)...")
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"  ✓ Variance expliquée: {pca.explained_variance_ratio_.sum():.4f}")
print(f"  ✓ Nouvelles dimensions: {X_train_pca.shape}")

# ============= 3. VALIDATION CROISÉE STRATIFIÉE =============
print("\n" + "=" * 60)
print("3️⃣  VALIDATION CROISÉE STRATIFIÉE (5-Fold)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modèle 1: RandomForest AMÉLIORÉ
print("\n[Model 1] RandomForest (tuné)...")
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

scoring = {'f1_weighted': 'f1_weighted', 'accuracy': 'accuracy'}
rf_cv = cross_validate(rf_clf, X_train_pca, y_train, cv=skf, scoring=scoring)

print(f"  F1-Score CV: {rf_cv['test_f1_weighted'].mean():.4f} (+/- {rf_cv['test_f1_weighted'].std():.4f})")
print(f"  Accuracy CV: {rf_cv['test_accuracy'].mean():.4f} (+/- {rf_cv['test_accuracy'].std():.4f})")

# Modèle 2: GradientBoosting AMÉLIORÉ
print("\n[Model 2] GradientBoosting (tuné)...")
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=5,
    random_state=42
)

gb_cv = cross_validate(gb_clf, X_train_pca, y_train, cv=skf, scoring=scoring)

print(f"  F1-Score CV: {gb_cv['test_f1_weighted'].mean():.4f} (+/- {gb_cv['test_f1_weighted'].std():.4f})")
print(f"  Accuracy CV: {gb_cv['test_accuracy'].mean():.4f} (+/- {gb_cv['test_accuracy'].std():.4f})")

# ============= 4. ENTRAÎNEMENT FINAL =============
print("\n" + "=" * 60)
print("4️⃣  ENTRAÎNEMENT FINAL")
print("=" * 60)

# Choisir le meilleur modèle basé sur CV
if rf_cv['test_f1_weighted'].mean() > gb_cv['test_f1_weighted'].mean():
    print("  → RandomForest sélectionné (meilleur CV F1)")
    best_clf = rf_clf
    best_name = "RandomForest"
else:
    print("  → GradientBoosting sélectionné (meilleur CV F1)")
    best_clf = gb_clf
    best_name = "GradientBoosting"

best_clf.fit(X_train_pca, y_train)

# ============= 5. ÉVALUATION TEST =============
print("\n" + "=" * 60)
print("5️⃣  ÉVALUATION SUR LE TEST SET")
print("=" * 60)

y_pred_test = best_clf.predict(X_test_pca)
y_pred_train = best_clf.predict(X_train_pca)

test_f1 = f1_score(y_test, y_pred_test, average='weighted')
train_f1 = f1_score(y_train, y_pred_train, average='weighted')

print(f"✓ Train F1-Score: {train_f1:.4f}")
print(f"✓ Test F1-Score: {test_f1:.4f}")

target_names = ['Baseline', 'Stress', 'Amusement']
print("\n📊 Rapport de Classification (Test):")
print(classification_report(y_test, y_pred_test, target_names=target_names))

# ============= 6. VISUALISATIONS =============
print("\n" + "=" * 60)
print("6️⃣  SAUVEGARDE DES RÉSULTATS")
print("=" * 60)

# Matrice de confusion
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=target_names, yticklabels=target_names)
axes[0].set_title(f'Test Set (F1: {test_f1:.4f})')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

cm_train = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=target_names, yticklabels=target_names)
axes[1].set_title(f'Train Set (F1: {train_f1:.4f})')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('saved_models/confusion_matrices_improved.png', dpi=100)
print("✓ Matrices de confusion sauvegardées")

# Résultats JSON
results = {
    'model': best_name,
    'cv_f1': rf_cv['test_f1_weighted'].mean() if best_name == "RandomForest" else gb_cv['test_f1_weighted'].mean(),
    'test_f1': float(test_f1),
    'train_f1': float(train_f1),
    'preprocessing': {
        'standardscaler': True,
        'pca_n_components': 256,
        'variance_explained': float(pca.explained_variance_ratio_.sum())
    }
}

with open('saved_models/training_results_improved.json', 'w') as f:
    json.dump(results, f, indent=4)

# Sauvegarder le modèle, scaler et PCA
joblib.dump(best_clf, f'saved_models/wesad_{best_name}_improved.joblib')
joblib.dump(scaler, 'saved_models/scaler.joblib')
joblib.dump(pca, 'saved_models/pca.joblib')

print(f"✓ Modèle sauvegardé: wesad_{best_name}_improved.joblib")
print(f"✓ Résultats: {results}")

if test_f1 >= 0.75:
    print("\n🎉 SUCCÈS! F1-Score >= 0.75")
else:
    print(f"\n⚠️  F1-Score actuel: {test_f1:.4f} (objectif: 0.80)")
