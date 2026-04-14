import torch
import numpy as np
from collections import deque
from shimmer_reader import stream_shimmer  # ton acquisition BT
from NormWear.main_model import NormWearModel

# ── Config ────────────────────────────────────────────────
WEIGHT_PATH = "normwear_checkpoint.pth"
SAMPLING_RATE = 64
WINDOW_SECONDS = 10          # fenêtre glissante
STEP_SECONDS   = 2           # mise à jour toutes les 2s
N_CHANNELS     = 2           # GSR + PPG
WINDOW_LEN     = SAMPLING_RATE * WINDOW_SECONDS

# ── Modèle ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NormWearModel(weight_path=WEIGHT_PATH, optimized_cwt=True).to(device)
model.eval()

# ── Buffer circulaire ─────────────────────────────────────
buffer = deque(maxlen=WINDOW_LEN)

def preprocess(window: np.ndarray) -> torch.Tensor:
    """window: [N_CHANNELS, WINDOW_LEN]"""
    # Normalisation z-score par canal
    mean = window.mean(axis=1, keepdims=True)
    std  = window.std(axis=1, keepdims=True) + 1e-8
    norm = (window - mean) / std
    return torch.tensor(norm, dtype=torch.float32).unsqueeze(0).to(device)

def get_fatigue_score(x: torch.Tensor) -> float:
    """Retourne un score de fatigue entre 0 (alerte) et 1 (fatigué)."""
    with torch.no_grad():
        emb = model.get_embedding(x, sampling_rate=SAMPLING_RATE, device=device)
        # Mean pooling sur patches + canaux → [768]
        feat = emb.mean(dim=2).mean(dim=1).squeeze(0)
    # → remplace ici par ton classifieur si tu as fine-tuné
    # Pour la démo : proxy simple sur la norme de l'embedding
    return float(torch.sigmoid(feat.norm() - 30).cpu())

# ── Boucle temps réel ─────────────────────────────────────
step_samples = SAMPLING_RATE * STEP_SECONDS
sample_count = 0

for sample in stream_shimmer(port="/dev/rfcomm0"):
    # sample = [gsr, ppg] (un point à 64 Hz)
    buffer.append(sample)
    sample_count += 1

    if len(buffer) == WINDOW_LEN and sample_count % step_samples == 0:
        window = np.array(buffer).T  # [2, WINDOW_LEN]
        x = preprocess(window)
        score = get_fatigue_score(x)
        label = "🔴 FATIGUÉ" if score > 0.6 else "🟡 MODÉRÉ" if score > 0.35 else "🟢 ALERTE"
        print(f"Score fatigue : {score:.3f}  →  {label}")