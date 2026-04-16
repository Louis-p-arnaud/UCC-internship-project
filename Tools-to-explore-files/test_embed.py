import pickle
import os
import numpy as np

file_path = r"/Data/wearable_downstream/wesad/sample_for_downstream/2_0_0_1563"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Clés disponibles :", data.keys())
print("Taille de l'embedding :", data['embed'].shape)
print("Labels associés :", data['label'])


def inspect_channels(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} est introuvable.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print("--- Analyse du fichier d'embedding ---")
    print(f"Fichier : {os.path.basename(file_path)}")

    # 1. Vérification de la forme de l'embedding
    # NormWear sort souvent : [Batch, Canaux, Patches, Dimension]
    emb = data['embed']
    print(f"Forme de l'embedding (Shape) : {emb.shape}")

    # 2. Extraction des infos de labels/canaux
    # Dans WESAD version NormWear, les infos de capteurs sont souvent
    # stockées dans la structure 'label' ou 'uid'
    if 'label' in data:
        print("\nContenu de la clé 'label' :")
        for i, task in enumerate(data['label']):
            print(f"  Index {i} : {task}")

    # 3. Test de cohérence statistique
    # Le GSR et le PPG ont des amplitudes et des variances très différentes.
    # Si 'data' brut est présent (rare dans l'embedding, mais possible selon votre script)
    if 'data' in data:
        sig = data['data']
        for ch in range(sig.shape[0]):
            print(f"\nStatistiques Canal {ch} :")
            print(f"  Moyenne : {np.mean(sig[ch]):.4f}")
            print(f"  Variance : {np.var(sig[ch]):.4f}")
            # Indice : Le PPG (cardiaque) a une variance beaucoup plus élevée
            # et cyclique que le GSR (lent) après normalisation.

inspect_channels(file_path)

if __name__ == "__main__":
    inspect_channels(file_path)