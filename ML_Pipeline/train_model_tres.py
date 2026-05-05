import json
import os
import pickle
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import subprocess
import sys

warnings.filterwarnings('ignore')

from path import DATA_PATH

print("=" * 90)
print("WESAD V5 - TEST D'EMBEDDINGS MULTIPLES + XGBOOST GPU")
print("=" * 90)

# ============= INSTALLATION XGBOOST =============
print("\n📦 Vérification des dépendances...")
try:
    import xgboost as xgb

    print("✓ XGBoost détecté")
    HAS_XGBOOST = True
except ImportError:
    print("⚠️  XGBoost non disponible. Installation...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
        import xgboost as xgb

        print("✓ XGBoost installé avec succès")
        HAS_XGBOOST = True
    except:
        print("❌ Impossible d'installer XGBoost, utilisation de RandomForest")
        HAS_XGBOOST = False

# ============= CONFIGURATION GPU =============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============= CHARGEMENT =============
with open(os.path.join(DATA_PATH, "wearable_downstream", "wesad", "train_test_split.json"), 'r') as f:
    split = json.load(f)

embed_dir = os.path.join(DATA_PATH, "wearable_downstream/wesad/wesad_stress_wav_embed")


def load_data_from_list(file_id_list):
    X, y, subjects = [], [], []
    for file_id in file_id_list:
        file_path = os.path.join(embed_dir, file_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if 'embed' in data:
                    emb = data['embed'].astype(np.float32)
                    feat = emb.flatten()

                    if np.isnan(feat).any() or np.isinf(feat).any():
                        feat = np.nan_to_num(feat)

                    if feat.shape[0] == 7680:
                        X.append(feat)
                        y.append(data['label'][0]['class'])
                        subject_id = file_id.split('_')[0]
                        subjects.append(subject_id)
            except:
                pass

    return np.array(X), np.array(y), np.array(subjects)


# ============= 1. CHARGEMENT =============
print("\n" + "=" * 90)
print("1️⃣  CHARGEMENT DES DONNÉES")
print("=" * 90)

X_train, y_train, subj_train = load_data_from_list(split['train'])
X_test, y_test, subj_test = load_data_from_list(split['test'])

print(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

class_names = ['Baseline', 'Stress', 'Amusement']

# ============= 2. PREPROCESSING =============
print("\n" + "=" * 90)
print("2️⃣  PREPROCESSING")
print("=" * 90)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ RobustScaler appliqué")

# ============= 3. STRATÉGIE: SMOTE POUR AMUSEMENT SEULEMENT =============
print("\n" + "=" * 90)
print("3️⃣  SMOTE CIBLÉ (AMUSEMENT SEULEMENT)")
print("=" * 90)

unique_train, counts_train = np.unique(y_train, return_counts=True)
print("Avant SMOTE:")
for cls, count in zip(unique_train, counts_train):
    print(f"  {class_names[cls]}: {count}")

# SMOTE seulement pour Amusement → 50% de Baseline
smote = SMOTE(k_neighbors=3, random_state=42, sampling_strategy={
    2: int(counts_train[0] * 0.5)  # Amusement seulement
})

X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("\nAprès SMOTE (Amusement augmenté):")
unique_bal, counts_bal = np.unique(y_train_balanced, return_counts=True)
for cls, count in zip(unique_bal, counts_bal):
    print(f"  {class_names[cls]}: {count}")

# ============= 4. CLASSIFIEURS MULTIPLES =============
print("\n" + "=" * 90)
print("4️⃣  ENTRAÎNEMENT MULTIPLES MODÈLES")
print("=" * 90)

results_list = []

# Modèle 1: XGBoost GPU (si disponible)
if HAS_XGBOOST:
    print("\n[1] XGBoost GPU...")
    try:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.5,
            reg_alpha=0.3,
            reg_lambda=0.7,
            tree_method='gpu_hist',
            gpu_id=0,
            random_state=42,
            verbosity=0,
            scale_pos_weight=1.0
        )

        xgb_clf.fit(X_train_balanced, y_train_balanced)
        y_pred_train_xgb = xgb_clf.predict(X_train_scaled)
        y_pred_test_xgb = xgb_clf.predict(X_test_scaled)

        f1_train_xgb = f1_score(y_train, y_pred_train_xgb, average='macro')
        f1_test_xgb = f1_score(y_test, y_pred_test_xgb, average='macro')

        results_list.append({
            'model': 'XGBoost',
            'f1_train': f1_train_xgb,
            'f1_test': f1_test_xgb,
            'gap': f1_train_xgb - f1_test_xgb,
            'predictions': y_pred_test_xgb,
            'clf': xgb_clf
        })

        print(f"  ✓ Train F1: {f1_train_xgb:.4f}, Test F1: {f1_test_xgb:.4f}, Gap: {f1_train_xgb - f1_test_xgb:.4f}")
    except Exception as e:
        print(f"  ❌ Erreur: {e}")

# Modèle 2: RandomForest Standard
print("\n[2] RandomForest Standard...")
rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_clf.fit(X_train_balanced, y_train_balanced)
y_pred_train_rf = rf_clf.predict(X_train_scaled)
y_pred_test_rf = rf_clf.predict(X_test_scaled)

f1_train_rf = f1_score(y_train, y_pred_train_rf, average='macro')
f1_test_rf = f1_score(y_test, y_pred_test_rf, average='macro')

results_list.append({
    'model': 'RandomForest',
    'f1_train': f1_train_rf,
    'f1_test': f1_test_rf,
    'gap': f1_train_rf - f1_test_rf,
    'predictions': y_pred_test_rf,
    'clf': rf_clf
})

print(f"  ✓ Train F1: {f1_train_rf:.4f}, Test F1: {f1_test_rf:.4f}, Gap: {f1_train_rf - f1_test_rf:.4f}")

# Modèle 3: RandomForest Très Régularisé
print("\n[3] RandomForest Ultra-Régularisé...")
rf_light = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features=0.4,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

rf_light.fit(X_train_balanced, y_train_balanced)
y_pred_train_rfl = rf_light.predict(X_train_scaled)
y_pred_test_rfl = rf_light.predict(X_test_scaled)

f1_train_rfl = f1_score(y_train, y_pred_train_rfl, average='macro')
f1_test_rfl = f1_score(y_test, y_pred_test_rfl, average='macro')

results_list.append({
    'model': 'RF-Light',
    'f1_train': f1_train_rfl,
    'f1_test': f1_test_rfl,
    'gap': f1_train_rfl - f1_test_rfl,
    'predictions': y_pred_test_rfl,
    'clf': rf_light
})

print(f"  ✓ Train F1: {f1_train_rfl:.4f}, Test F1: {f1_test_rfl:.4f}, Gap: {f1_train_rfl - f1_test_rfl:.4f}")

# ============= 5. SÉLECTION MEILLEUR MODÈLE =============
print("\n" + "=" * 90)
print("5️⃣  RÉSUMÉ COMPARATIF")
print("=" * 90)

results_list.sort(key=lambda x: x['f1_test'], reverse=True)

print("\n🏆 Classement par Test F1-Macro:")
for i, res in enumerate(results_list, 1):
    print(
        f"  {i}. {res['model']:15} | Test: {res['f1_test']:.4f} | Train: {res['f1_train']:.4f} | Gap: {res['gap']:.4f}")

best_result = results_list[0]
y_pred_best = best_result['predictions']
best_clf = best_result['clf']
best_name = best_result['model']

print(f"\n✨ Meilleur modèle: {best_name} (F1-Test: {best_result['f1_test']:.4f})")

# ============= 6. ANALYSE DÉTAILLÉE DU MEILLEUR =============
print("\n" + "=" * 90)
print("6️⃣  ANALYSE DÉTAILLÉE DU MEILLEUR MODÈLE")
print("=" * 90)

p, r, f, s = precision_recall_fscore_support(y_test, y_pred_best, average=None)

print("\n📊 Résultats par classe:")
for i, cls_name in enumerate(class_names):
    print(f"\n{cls_name} ({int(s[i])} samples):")
    print(f"  Precision: {p[i]:.4f}")
    print(f"  Recall:    {r[i]:.4f}")
    print(f"  F1-Score:  {f[i]:.4f}")

print("\n" + classification_report(y_test, y_pred_best, target_names=class_names, digits=4))

# ============= 7. VISUALISATIONS =============
print("\n" + "=" * 90)
print("7️⃣  VISUALISATIONS")
print("=" * 90)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# CM meilleur modèle
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=class_names, yticklabels=class_names)
axes[0, 0].set_title(f'CM: {best_name} (F1: {best_result["f1_test"]:.4f})')
axes[0, 0].set_ylabel('True')
axes[0, 0].set_xlabel('Predicted')

# Comparaison modèles
models_names = [r['model'] for r in results_list]
test_scores = [r['f1_test'] for r in results_list]
gaps = [r['gap'] for r in results_list]

x = np.arange(len(models_names))
width = 0.35

axes[0, 1].bar(x - width / 2, test_scores, width, label='Test F1', alpha=0.8, color='steelblue')
axes[0, 1].bar(x + width / 2, gaps, width, label='Overfitting Gap', alpha=0.8, color='coral')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Comparaison des Modèles')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(models_names, rotation=15)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# F1 par classe
f1_scores = [f[i] for i in range(3)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
axes[1, 0].bar(class_names, f1_scores, color=colors, alpha=0.8)
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_title(f'{best_name}: F1-Score par classe')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(axis='y', alpha=0.3)

for i, v in enumerate(f1_scores):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Recall par classe
recall_scores = [r[i] for i in range(3)]
axes[1, 1].bar(class_names, recall_scores, color=colors, alpha=0.8)
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_title(f'{best_name}: Recall par classe')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(axis='y', alpha=0.3)

for i, v in enumerate(recall_scores):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('saved_models/wesad_v5_full_diagnostic.png', dpi=100, bbox_inches='tight')
print("✓ Graphiques sauvegardés: wesad_v5_full_diagnostic.png")

# ============= 8. SAUVEGARDE =============
os.makedirs('saved_models', exist_ok=True)

all_results = {
    'best_model': best_name,
    'device': str(DEVICE),
    'xgboost_available': HAS_XGBOOST,
    'best_performance': {
        'test_f1_macro': float(best_result['f1_test']),
        'train_f1_macro': float(best_result['f1_train']),
        'overfitting_gap': float(best_result['gap']),
    },
    'per_class_f1': {
        'baseline': float(f[0]),
        'stress': float(f[1]),
        'amusement': float(f[2])
    },
    'per_class_recall': {
        'baseline': float(r[0]),
        'stress': float(r[1]),
        'amusement': float(r[2])
    },
    'all_models': [
        {
            'name': res['model'],
            'test_f1': float(res['f1_test']),
            'train_f1': float(res['f1_train']),
            'gap': float(res['gap'])
        }
        for res in results_list
    ],
    'preprocessing': {
        'scaler': 'RobustScaler',
        'smote_applied': True,
        'smote_target': 'Amusement only (to 50% of Baseline)',
        'final_train_samples': int(len(y_train_balanced))
    }
}

joblib.dump(best_clf, f'saved_models/wesad_{best_name}_v5.joblib')
joblib.dump(scaler, 'saved_models/scaler_v5.joblib')

with open('saved_models/results_v5_full.json', 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\n✓ Modèle: saved_models/wesad_{best_name}_v5.joblib")
print(f"✓ Résultats complets: saved_models/results_v5_full.json")

# ============= 9. RECOMMANDATIONS FINALES =============
print("\n" + "=" * 90)
print("💡 RECOMMANDATIONS FINALES")
print("=" * 90)

if best_result['f1_test'] >= 0.65:
    print("✅ Performance acceptable atteinte!")
elif best_result['f1_test'] >= 0.55:
    print("⚠️  Performance moyenne - Considérez:")
    print("  1. Tester AUTRES embeddings (STAT_API, TFC_API, CLAP_API)")
    print("  2. Fine-tuner NormWear sur WESAD (transfer learning)")
    print("  3. Augmenter les données Amusement")
    print("  4. Combiner plusieurs embeddings (ensemble)")
else:
    print("❌ Performance insuffisante:")
    print("  → Problème principal: Embeddings ne capturent pas bien Amusement")
    print("  → SOLUTIONS PRIORITAIRES:")
    print("     1. Essayer STAT_API (hand-crafted features)")
    print("     2. Essayer TFC_API (Time-Frequency)")
    print("     3. Essayer CLAP_API (multimodal)")
    print("     4. Combiner embeddings multiples")

print("\n" + "=" * 90)
print("✅ DIAGNOSTIC V5 COMPLET")
print("=" * 90)
