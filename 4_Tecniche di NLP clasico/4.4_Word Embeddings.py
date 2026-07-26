import os

# --- CONFIGURAZIONE AMBIENTE ---
# Configurazione del Backend per Keras 3 (Standard 2026)
# INTERAZIONE: Keras 3 è agnostico rispetto al backend. Impostando "torch", istruiamo 
# Keras a utilizzare PyTorch per gestire i calcoli tensoriali e i grafi di computazione.
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import ops # Keras Ops: operazioni matematiche che funzionano su tutti i backend (Torch, JAX, TF)

# --- 1. DEFINIZIONE DEL VOCABOLARIO E TOKENIZZAZIONE ---
# INTERAZIONE: Iniziamo trasformando il testo in numeri. Ogni parola è mappata a un ID intero.
# Questo è il primo passo per trasformare concetti astratti in dati leggibili dalla macchina.
vocab = {"gatto": 0, "cane": 1, "astronave": 2}
dim_voc = len(vocab)
embedding_dim = 2  # Dimensione dello spazio semantico (2D per facilità di comprensione)

# --- 2. IL LIMITE DELLA RAPPRESENTAZIONE ONE-HOT ---
# TEORIA: I vettori One-Hot sono "ortogonali" (perpendicolari). 
# Questo significa che il prodotto scalare tra due parole diverse sarà sempre 0.
# LIMITE: Il sistema non può capire che "cane" è più vicino a "gatto" rispetto a "astronave".
def get_one_hot(word_id, size):
    """Crea un vettore di zeri con un unico '1' nella posizione dell'ID parola."""
    vec = np.zeros(size)
    vec[word_id] = 1
    return vec

# Estraiamo i vettori One-Hot per il confronto
v_gatto_oh = get_one_hot(vocab["gatto"], dim_voc) # [1, 0, 0]
v_cane_oh = get_one_hot(vocab["cane"], dim_voc)   # [0, 1, 0]

# Calcolo Prodotto Scalare One-Hot: Risultato 0 -> Zero similarità matematica nonostante la vicinanza semantica.
print(f"Similarità scalare One-Hot (Gatto-Cane): {np.dot(v_gatto_oh, v_cane_oh)}")

# --- 3. WORD EMBEDDINGS (Rappresentazione Densa e Continua) ---
# INTERAZIONE: Il layer 'Embedding' agisce come una tabella di lookup gigante.
# Invece di vettori sparsi (0 e 1), usa coordinate reali (es. 0.9, 0.1).
embedding_layer = keras.layers.Embedding(input_dim=dim_voc, output_dim=embedding_dim)

# Inizializziamo i pesi manualmente per simulare un addestramento avvenuto.
# NOTA: Gatto e Cane hanno coordinate molto vicine nello spazio 2D, Astronave è su un altro quadrante.
custom_weights = np.array([
    [0.9, 0.1],  # Coordinata Gatto
    [0.8, 0.2],  # Coordinata Cane (Vicina a Gatto)
    [-0.5, 0.8]   # Coordinata Astronave (Lontana dagli animali)
])

# Costruiamo il layer e iniettiamo i nostri pesi personalizzati
embedding_layer.build()
embedding_layer.set_weights([custom_weights])

# --- 4. CALCOLO DELLA SIMILARITÀ SEMANTICA (Cosine Similarity) ---
# INTERAZIONE: La similarità coseno misura l'angolo tra due vettori. 
# Più l'angolo è piccolo, più il valore si avvicina a 1 (massima similarità).
def cosine_similarity(v1, v2):
    # Usiamo 'keras.ops' per garantire che la funzione funzioni indipendentemente dal backend scelto (Torch/TF)
    dot_product = ops.sum(v1 * v2)
    norm_v1 = ops.sqrt(ops.sum(ops.square(v1)))
    norm_v2 = ops.sqrt(ops.sum(ops.square(v2)))
    # Formula: (A dot B) / (||A|| * ||B||)
    return dot_product / (norm_v1 * norm_v2)

# PASSAGGIO CHIAVE: Trasformiamo gli ID in vettori densi passando attraverso il layer di Embedding
# 'embedding_layer' riceve un input numerico e restituisce il vettore di 'embedding_dim' componenti.
vec_gatto = embedding_layer(np.array([vocab["gatto"]]))
vec_cane = embedding_layer(np.array([vocab["cane"]]))
vec_astronave = embedding_layer(np.array([vocab["astronave"]]))

# Calcoliamo le distanze semantiche
sim_animali = cosine_similarity(vec_gatto, vec_cane)
sim_misto = cosine_similarity(vec_gatto, vec_astronave)

print(f"\nSimilarità Coseno (Gatto - Cane): {float(sim_animali):.4f} (Alta: sono entrambi animali)")
print(f"Similarità Coseno (Gatto - Astronave): {float(sim_misto):.4f} (Bassa: concetti distanti)")

# --- 5. PERSISTENZA DEI DATI ---
# Creiamo un modello 'Sequential' che incapsula il nostro layer.
# Questo permette di salvare non solo i pesi, ma l'intera architettura del sistema di embedding.
model = keras.Sequential([embedding_layer])
model.save("word_embeddings_2026.keras")
print("\n[INFO] Modello salvato: la mappa semantica è ora pronta per essere riutilizzata.")