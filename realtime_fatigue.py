import torch
import numpy as np
from collections import deque
import time

# --- Section désactivée pour le test (acquisition BT) ---
# from shimmer_reader import stream_shimmer
# --------------------------------------------------------

# Si tu n'as pas encore le fichier main_model localement,
# assure-toi que le dossier NormWear est dans ton PYTHONPATH.
from NormWear.main_model import NormWearModel

# ── Config ────────────────────────────────────────────────
WEIGHT_PATH = "normwear_checkpoint.pth"
SAMPLING_RATE = 64
WINDOW_SECONDS = 10  # Fenêtre glissante de 10s
STEP_SECONDS = 2  # Mise à jour toutes les 2s
N_CHANNELS = 2  # GSR + PPG
WINDOW_LEN = SAMPLING_RATE * WINDOW_SECONDS

# ── Modèle ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note: Assure-toi que le poids existe, sinon initialise un modèle vierge pour le test
try:
    model = NormWearModel(weight_path=WEIGHT_PATH, optimized_cwt=True).to(device)
    print(f"Modèle chargé sur {device}")
except Exception as e:
    print(f"Erreur chargement poids : {e}. Initialisation d'un modèle aléatoire pour test.")
    model = NormWearModel(optimized_cwt=True).to(device)

model.eval()

# ── Buffer circulaire ─────────────────────────────────────
buffer = deque(maxlen=WINDOW_LEN)


def preprocess(window: np.ndarray) -> torch.Tensor:
    """window: [N_CHANNELS, WINDOW_LEN]"""
    # Normalisation z-score par canal
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    norm = (window - mean) / std
    # Format attendu par le modèle : [Batch, Channels, Time]
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)


def get_fatigue_score(x: torch.Tensor) -> float:
    """Retourne un score de fatigue via l'embedding."""
    with torch.no_grad():
        # Extraction de l'embedding
        emb = model.get_embedding(x, sampling_rate=SAMPLING_RATE, device=device)

        # Mean pooling pour obtenir un vecteur fixe (ex: taille 768)
        # On réduit les dimensions temporelles et spatiales
        feat = emb.mean(dim=2).mean(dim=1).squeeze(0)

    # --- ICI : Ton futur classifieur simple (ex: Arbre de Décision) ---
    # Pour l'instant, on garde ta logique de proxy via la norme du vecteur
    score = torch.sigmoid(feat.norm() - 30)
    return float(score.cpu())


# ── Boucle temps réel simulée ──────────────────────────────
step_samples = SAMPLING_RATE * STEP_SECONDS
sample_count = 0

print("Début de la simulation (Signal aléatoire)...")

try:
    # --- Remplacement de stream_shimmer par un générateur aléatoire ---
    # for sample in stream_shimmer(port="/dev/rfcomm0"):
    while True:
        # Simulation d'un échantillon [GSR, PPG]
        sample = np.random.randn(N_CHANNELS)

        buffer.append(sample)
        sample_count += 1

        # On attend d'avoir assez de données pour la première fenêtre
        if len(buffer) == WINDOW_LEN and sample_count % step_samples == 0:
            window = np.array(buffer).T  # Shape [2, 640]
            x = preprocess(window)

            score = get_fatigue_score(x)

            label = "🔴 FATIGUÉ" if score > 0.6 else "🟡 MODÉRÉ" if score > 0.35 else "🟢 ALERTE"

            print(f"[{time.strftime('%H:%M:%S')}] Score fatigue : {score:.3f}  →  {label}")

        # Simuler la fréquence d'échantillonnage (1/64 Hz)
        time.sleep(1 / SAMPLING_RATE)

except KeyboardInterrupt:
    print("\nArrêt de la simulation.")