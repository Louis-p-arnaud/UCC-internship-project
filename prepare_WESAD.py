import pickle
import numpy as np
import json
import os
from path import PROJECT_PATH, WESAD_PATH, DATA_PATH

OUTPUT_PATH = os.path.join(DATA_PATH,"wesad_normwear")
WINDOW_SEC  = 10
STEP_SEC    = 5

# Fréquences d'échantillonnage WESAD (capteur poignet E4)
SR = {
    "BVP": 64,   # PPG
    "EDA": 4,    # GSR
    "ACC": 32,
    "TEMP": 4,
}

# On utilise BVP (64 Hz) comme référence - on resample EDA/TEMP à 64 Hz
TARGET_SR = 64

def resample(signal, from_sr, to_sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(to_sr), int(from_sr))
    return resample_poly(signal, int(to_sr)//g, int(from_sr)//g)

def convert_subject(subject_id: str):
    pkl_path = os.path.join(WESAD_PATH, subject_id, f"{subject_id}.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    wrist = data["signal"]["wrist"]
    labels_raw = data["label"]  # 700 Hz

    # Signaux poignet
    bvp  = wrist["BVP"].flatten()          # 64 Hz
    eda  = resample(wrist["EDA"].flatten(), SR["EDA"],  TARGET_SR)
    temp = resample(wrist["TEMP"].flatten(), SR["TEMP"], TARGET_SR)
    acc  = wrist["ACC"]  # [N, 3] à 32 Hz → resample
    acc_r = np.stack([resample(acc[:, i], SR["ACC"], TARGET_SR) for i in range(3)])

    # Aligne les longueurs sur le plus court
    min_len = min(len(bvp), len(eda), len(temp), acc_r.shape[1])
    signal_matrix = np.stack([
        bvp[:min_len],
        eda[:min_len],
        temp[:min_len],
        acc_r[0, :min_len],
        acc_r[1, :min_len],
        acc_r[2, :min_len],
    ])  # [6, min_len]

    # Resample les labels (700 Hz → 64 Hz)
    label_resampled = resample(labels_raw.astype(float), 700, TARGET_SR)
    label_resampled = np.round(label_resampled).astype(int)[:min_len]

    # Fenêtrage glissant
    win = TARGET_SR * WINDOW_SEC
    step = TARGET_SR * STEP_SEC
    samples = []

    for start in range(0, min_len - win, step):
        end = start + win
        window_labels = label_resampled[start:end]
        # Label majoritaire dans la fenêtre (ignore 0 = transient)
        valid = window_labels[window_labels > 0]
        if len(valid) < win * 0.8:
            continue  # fenêtre trop bruitée
        majority_label = int(np.bincount(valid).argmax())
        if majority_label == 0:
            continue

        # Convertit label WESAD en classe fatigue/stress
        # 1=baseline→0(calme), 2=stress→1(stress), 3=amusement→0(calme)
        fatigue_label = 1 if majority_label == 2 else 0

        sample = {
            "uid": subject_id,
            "data": signal_matrix[:, start:end].astype(np.float16),
            "sampling_rate": TARGET_SR,
            "label": [{"class": fatigue_label}]
        }
        fname = f"{subject_id}_w{start}.pkl"
        out_dir = os.path.join(OUTPUT_PATH, "sample_for_downstream")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, fname), "wb") as f:
            pickle.dump(sample, f)
        samples.append(fname)

    return samples

# ── Traitement de tous les sujets ────────────────────────
all_samples = []
subjects = [f"S{i}" for i in range(2, 18) if i != 12]  # S12 manquant dans WESAD

for s in subjects:
    path = os.path.join(WESAD_PATH, s)
    if os.path.exists(path):
        print(f"Traitement {s}...")
        all_samples += [(s, f) for f in convert_subject(s)]

# ── Train/test split (leave-one-subject-out) ─────────────
from sklearn.model_selection import GroupShuffleSplit
import random

all_files  = [f for _, f in all_samples]
all_groups = [s for s, _ in all_samples]

random.seed(42)
test_subjects  = random.sample(subjects[:len(subjects)], k=3)
train_files = [f for s, f in all_samples if s not in test_subjects]
test_files  = [f for s, f in all_samples if s in test_subjects]

split = {"train": train_files, "test": test_files}
with open(os.path.join(OUTPUT_PATH, "train_test_split.json"), "w") as f:
    json.dump(split, f)

print(f"Done — {len(train_files)} train, {len(test_files)} test samples")
print(f"Sujets test : {test_subjects}")