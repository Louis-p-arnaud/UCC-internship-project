import torch
import numpy as np
from collections import deque
import time
import joblib

# ── Configuration ─────────────────────────────────────────
WEIGHT_PATH = "normwear_checkpoint.pth"
SAMPLING_RATE = 64
WINDOW_SECONDS = 10
STEP_SECONDS = 2
WINDOW_LEN = SAMPLING_RATE * WINDOW_SECONDS

# Mapping identifié : NormWear attend 10 canaux
N_CHANNELS_MODEL = 10
CH_PPG = 5  # Index du PPG selon l'analyse
CH_GSR = 6  # Index du GSR selon l'analyse

LABELS_MAP = {
    0: "🟢 NEUTRE (Repos)",
    1: "🔴 STRESS (Alerte)",
    2: "🔵 AMUSEMENT (Actif)"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Chargement des Modèles ────────────────────────────────
try:
    clf = joblib.load('fatigue_tree.joblib')
    from NormWear.main_model import NormWearModel

    model = NormWearModel(weight_path=WEIGHT_PATH, optimized_cwt=True).to(device)
    model.eval()
    print(f"Système prêt sur {device} (Entrée : 10 canaux)")
except Exception as e:
    print(f"Erreur initialisation : {e}")
    exit()

# ── Buffer Circulaire ─────────────────────────────────────
# Le buffer stocke maintenant les 10 canaux requis par le modèle
buffer = deque(maxlen=WINDOW_LEN)


def generate_simulated_shimmer_sample(t, fs):
    """
    Simule des signaux cohérents pour le Shimmer3.
    t : temps actuel en secondes
    """
    # 1. Simulation PPG (Canal 5) : Sinusoïde à ~1.2Hz (72 BPM) + une harmonique
    ppg = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)

    # 2. Simulation GSR (Canal 6) : Dérive très lente (Random Walk)
    # Note: On utilise une petite composante aléatoire persistante
    gsr = 0.5 + 0.05 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.001)

    # Création du vecteur de 10 canaux (remplissage du reste avec du bruit léger)
    full_sample = np.random.normal(0, 0.01, N_CHANNELS_MODEL)
    full_sample[CH_PPG] = ppg
    full_sample[CH_GSR] = gsr

    return full_sample


def preprocess(window: np.ndarray) -> torch.Tensor:
    """window shape: [10, WINDOW_LEN]"""
    # Normalisation z-score par canal (crucial pour NormWear)
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    norm = (window - mean) / std
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)


def get_realtime_classification(x: torch.Tensor):
    with torch.no_grad():
        # 1. Obtenir l'embedding brut [Batch, Canaux, Patches, Dim]
        out = model.get_embedding(x, sampling_rate=SAMPLING_RATE, device=device)

        # 2. RÉDUIRE LA DIMENSION TEMPORELLE (Mean Pooling)
        # On passe de [1, 10, P, 768] à [1, 10, 768]
        # C'est cette étape qui permet de passer de 7 millions à 7680 caractéristiques
        embedding = out.mean(dim=2)

        # 3. Aplatir pour sklearn -> [1, 7680]
        feat_flat = embedding.cpu().numpy().flatten().reshape(1, -1)

        # 4. Prédire
        class_idx = clf.predict(feat_flat)[0]
        probs = clf.predict_proba(feat_flat)[0]
        return class_idx, probs[class_idx]

# ── Boucle de Simulation ──────────────────────────────────
step_samples = SAMPLING_RATE * STEP_SECONDS
sample_count = 0

print("\n--- Simulation Temps Réel Shimmer3 -> NormWear ---")
try:
    while True:
        t = sample_count / SAMPLING_RATE

        # Génération d'un échantillon simulé respectant le mapping
        sample = generate_simulated_shimmer_sample(t, SAMPLING_RATE)

        buffer.append(sample)
        sample_count += 1

        if len(buffer) == WINDOW_LEN and sample_count % step_samples == 0:
            # Conversion buffer [Time, Chan] -> [Chan, Time]
            window = np.array(buffer).T

            x_tensor = preprocess(window)
            class_id, conf = get_realtime_classification(x_tensor)

            status = LABELS_MAP.get(class_id, "Inconnu")
            print(
                f"[{time.strftime('%H:%M:%S')}] {status} | Confiance: {conf:.2f} | PPG(sync): {window[CH_PPG, -1]:.2f}")

        time.sleep(1 / SAMPLING_RATE)

except KeyboardInterrupt:
    print("\nSimulation arrêtée.")