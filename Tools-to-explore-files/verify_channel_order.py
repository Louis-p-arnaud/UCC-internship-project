import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, correlate

# --- Chargement ---
path = r"C:\Users\Louis\Documents\Louis-project\UCC-internship-project\Data\wearable_downstream\wesad\sample_for_downstream\2_0_1_3508"
with open(path, 'rb') as f:
    sample = pickle.load(f)

data = sample['data']
sr = sample['sampling_rate']

def get_bpm(signal, fs, distance_sec=0.5):
    # On normalise pour faciliter la détection de pics
    sig_norm = (signal - np.mean(signal)) / np.std(signal)
    # Distance minimale entre battements (ex: 0.5s = 120 BPM max)
    peaks, _ = find_peaks(sig_norm, distance=int(fs * distance_sec), height=1.0)
    if len(peaks) < 2: return 0
    intervals = np.diff(peaks) / fs
    bpm = 60 / np.mean(intervals)
    return bpm

print(f"--- ANALYSE DE VALIDATION (Sujet {sample.get('uid')}) ---")

# 1. Vérification du Rythme Cardiaque (C3 vs C5)
# Si C3=ECG et C5=PPG, les BPM doivent être quasi identiques.
bpm_c3 = get_bpm(data[3], sr)
bpm_c5 = get_bpm(data[5], sr)

print(f"BPM calculé sur Canal 3 (ECG supposé) : {bpm_c3:.2f}")
print(f"BPM calculé sur Canal 5 (PPG supposé) : {bpm_c5:.2f}")

if abs(bpm_c3 - bpm_c5) < 5:
    print("✅ VALIDÉ : Le Canal 3 et le Canal 5 partagent le même rythme cardiaque.")
else:
    print("❌ ALERTE : Rythmes cardiaques incohérents. Le mapping des capteurs cardiaques est peut-être différent.")

# 2. Corrélation GSR (C4 vs C6)
# Le GSR change lentement. On vérifie si la tendance est la même.
c4_norm = (data[4] - np.mean(data[4])) / np.std(data[4])
c6_norm = (data[6] - np.mean(data[6])) / np.std(data[6])
correlation = np.corrcoef(c4_norm, c6_norm)[0, 1]

print(f"Corrélation GSR (Canal 4 vs Canal 6) : {correlation:.3f}")
if correlation > 0.5:
    print("✅ VALIDÉ : Forte corrélation entre les canaux EDA/GSR.")
else:
    print("⚠️ WARNING : Faible corrélation. Possible que l'un soit de la température ou du bruit.")

# 3. Signature Accéléromètre (Statique vs Dynamique)
# On vérifie la variance. L'accéléromètre poignet (C7-9) bouge souvent plus que le torse (C0-2).
var_torse = np.mean([np.var(data[i]) for i in [0,1,2]])
var_poignet = np.mean([np.var(data[i]) for i in [7,8,9]])
print(f"Variance moyenne Torse (C0-2): {var_torse:.4f} | Poignet (C7-9): {var_poignet:.4f}")

# --- Affichage Final ---
fig, axs = plt.subplots(10, 1, figsize=(12, 18), sharex=True)
labels = ["ACC-X-T", "ACC-Y-T", "ACC-Z-T", "ECG", "GSR-T", "PPG", "GSR-P", "ACC-X-P", "ACC-Y-P", "ACC-Z-P"]

for i in range(10):
    axs[i].plot(data[i])
    axs[i].set_title(f"Canal {i}")#(f"Canal {i} - {labels[i]}")
    axs[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()