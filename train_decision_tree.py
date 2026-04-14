import json
import os
import torch
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from path import DATA_PATH

# 1. Charger le dictionnaire du split
with open(os.path.join(DATA_PATH,"wearable_downstream","wesad","train_test_split.json"), 'r') as f:
    split = json.load(f)

embed_dir = r"C:\Users\Louis\Documents\Louis-project\UCC-internship-project\Data\wearable_downstream\wesad\wesad_stress_wav_embed"



def load_data_from_list(file_id_list):
    X, y = [], []
    for file_id in file_id_list:
        file_path = os.path.join(embed_dir, f"{file_id}.pt")  # ou .npy
        if os.path.exists(file_path):
            # Charger l'embedding
            emb = torch.load(file_path, map_location='cpu')
            X.append(emb.detach().numpy().flatten())

            # Extraire le label depuis le file_id ou un CSV séparé
            # Exemple : si le 2ème chiffre est le label fatigue (0 ou 1)
            label = int(file_id.split('_')[1])
            y.append(label)
    return np.array(X), np.array(y)


# 2. Créer les jeux d'entraînement et de test réels
X_train, y_train = load_data_from_list(split['train'])
X_test, y_test = load_data_from_list(split['test'])

# 3. Entraîner l'arbre
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# 4. Sauvegarder pour ton script temps réel
joblib.dump(clf, 'fatigue_tree.joblib')