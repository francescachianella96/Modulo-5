import numpy as np

# 1. Documenti
docs = ["ai è il futuro", "il futuro è oggi"]

# 2. Creazione vocabolario unico
# Teoria: Il set garantisce l'unicità dei termini (Hapax legomena vs termini comuni)
vocab = sorted(list(set(" ".join(docs).split())))
word_idx = {w: i for i, w in enumerate(vocab)}

# 3. Costruzione Matrice DTM
matrix = np.zeros((len(docs), len(vocab)))
for i, d in enumerate(docs):
    for w in d.split():
        matrix[i, word_idx[w]] += 1

# 4. Calcolo Sparsità
# Formula: S = 1 - (Elementi_non_zero / Elementi_totali)
zeros = (matrix == 0).sum()
total = matrix.size
sparsity = zeros / total

print(f"Matrice:\n{matrix}")
print(f"Sparsità: {sparsity * 100:.2f}%")

# RISPOSTA ALLA DOMANDA TEORICA:
# Se aggiungessi un documento con 50 parole nuove, il vocabolario passerebbe da 5 a 55 termini.
# La riga del nuovo documento avrebbe 50 valori non nulli, ma le righe dei documenti vecchi 
# avrebbero 50 nuovi zeri ciascuna. La sparsità aumenterebbe drasticamente (sparsity -> 1.0).