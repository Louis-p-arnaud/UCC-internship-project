import json
import os
import torch
import pickle
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from path import DATA_PATH

# 1. Charger le dictionnaire du split
with open(os.path.join(DATA_PATH,"wearable_downstream","wesad","train_test_split.json"), 'r') as f:
    split = json.load(f)

embed_dir = r"C:\Users\Louis\Documents\Louis-project\UCC-internship-project\Data\wearable_downstream\wesad\wesad_stress_wav_embed"


def load_data_from_list(file_id_list):
    '''
    Charge les données du dossier d'embedding "sample_for_downstream"

    X (Features) : Pour chaque fichier, récupérez data['embed'].
    Comme c'est un modèle NormWear, l'embedding a souvent une forme multidimensionnelle $[1, nvar, P, E]$.
    On l'aplatit en un vecteur 1D pour l'arbre de décision (ex: embed.flatten()).

    y (Labels) : Récupérez la valeur numérique du label.
    Dans WESAD, c'est l'index de la classe (0, 1 ou 2).

    Exemple de structure d'un fichier downstream:
        Clés disponibles : dict_keys(['uid', 'sampling_rate', 'embed', 'label'])
        Taille de l'embedding : (7680,)
        Labels associés : [{'class': 1}]
    '''
    X, y = [], []
    for file_id in file_id_list:
        file_path = os.path.join(embed_dir, file_id)

        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                # Vérification de la structure attendue
                if 'embed' in data:
                    emb = data['embed'].astype(np.float32)

                    # SI vos fichiers d'entraînement n'étaient pas encore moyennés,
                    # il faudrait ajouter : if emb.ndim > 1: emb = emb.mean(axis=1)

                    feat = emb.flatten()

                    # Sécurité : on s'assure que chaque échantillon fait bien 7680
                    if feat.shape[0] == 7680:
                        X.append(feat)
                        label = data['label'][0]['class']
                        y.append(label)
                    else:
                        print(f"Format incorrect pour {file_id}: {feat.shape[0]}")

            except Exception as e:
                print(f"Erreur lors de la lecture de {file_id}: {e}")

    return np.array(X), np.array(y)

# 2. Créer les jeux d'entraînement et de test réels
print("creating training and test sets")
X_train, y_train = load_data_from_list(split['train']) # on charge tous les fichiers de train spécifiés dans le fichier train_test_split.json
X_test, y_test = load_data_from_list(split['test']) # idem pout les fichiers de test

# 3. Entraîner l'arbre
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# 4. Sauvegarder
joblib.dump(clf, 'fatigue_tree.joblib')
print("training complete, tree saved")