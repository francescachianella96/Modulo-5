"""
========================================================================================
WORD EMBEDDINGS E TRANSFER LEARNING REALE (KERAS 3 + PYTORCH)
========================================================================================
Obiettivo: Trasformare parole umane in vettori dotati di significato usando pesi GloVe reali.

INTERAZIONI CHIAVE:
1. TextVectorization: Il "Portinaio" che trasforma il testo grezzo in numeri (indici).
2. Gensim: Il "Fornitore" che scarica i vettori pre-addestrati da miliardi di documenti.
3. Embedding Matrix: Il "Ponte" che associa i nostri indici locali ai vettori di Gensim.
4. Embedding Layer: Il "Cuore" della rete che memorizza e proietta i vettori nello spazio.
========================================================================================
"""

import os

# FASE 0: SETUP DEL MOTORE DI CALCOLO
# Keras 3 è agnostico: qui scegliamo PyTorch come motore per le operazioni tensoriali.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

# Verifichiamo la presenza di Gensim (essenziale per scaricare i vettori GloVe reali)
try:
    import gensim.downloader as api
except ImportError:
    print("[ERRORE] Libreria 'gensim' mancante. Esegui: pip install gensim")
    exit()

def prepare_data():
    """
    FASE 1: PREPARAZIONE DEL TESTO E DEL VOCABOLARIO (Slide 2, 7, 8)
    ---------------------------------------------------------------
    Qui trasformiamo frasi di lunghezza diversa in una matrice numerica fissa.
    """
    # Piccoli esempi di sentiment analysis (Positivo vs Negativo)
    testi = [
        "The movie was absolutely fantastic and worth watching",
        "Terrible service and very bad food quality",
        "An incredible experience that I loved",
        "I did not like it at all, very boring"
    ]
    # Labels binarie: 1=Felice, 0=Triste
    labels = np.array([1, 0, 1, 0], dtype="float32")
    
    # TextVectorization (Il nostro sarto professionista):
    # - max_tokens: Considera solo le 1000 parole più frequenti.
    # - output_mode='int': Ogni parola diventa un intero unico.
    # - output_sequence_length=10: Fa PADDING (aggiunge zeri) o CLIP (taglia) a 10 parole.
    vectorizer = layers.TextVectorization(
        max_tokens=1000,
        output_mode="int",
        output_sequence_length=10, 
    )
    
    # 'adapt' legge i testi e costruisce internamente il vocabolario (mappa parola <-> indice)
    vectorizer.adapt(testi)
    
    return testi, labels, vectorizer

def build_real_transfer_model(vectorizer, embedding_dim=50):
    """
    FASE 2: COSTRUZIONE DEL MODELLO E TRASFUSIONE DI CONOSCENZA (Slide 11, 12, 13)
    -------------------------------------------------------------------------
    Qui carichiamo GloVe e iniettiamo i suoi pesi nel nostro layer Keras.
    """
    # Recuperiamo il vocabolario creato dal vectorizer
    vocab = vectorizer.get_vocabulary()
    num_tokens = len(vocab)
    
    # --- PASSO A: Caricamento Pesi GloVe (Conoscenza Esterna) ---
    print("\n[INFO] Download di 'glove-wiki-gigaword-50' (Conoscenza distillata da miliardi di parole)...")
    glove_vectors = api.load("glove-wiki-gigaword-50")
    
    # --- PASSO B: Creazione Matrice Ponte (La Trasfusione) ---
    # Creiamo una tabella vuota (zeri) di forma (NumeroParole x 50 Dimensioni)
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    
    hits, misses = 0, 0
    for i, word in enumerate(vocab):
        # Cerchiamo se la parola del nostro piccolo vocabolario esiste in GloVe
        if glove_vectors.has_index_for(word):
            # Se esiste, copiamo il vettore densi (il significato semantico)
            embedding_matrix[i] = glove_vectors[word]
            hits += 1
        else:
            # Se non esiste (es. nomi propri strani), la parola rimane con vettore zero
            misses += 1
            
    print(f"[INFO] Trasfusione: {hits} parole mappate, {misses} inizializzate a zero.")

    # --- PASSO C: Architettura del Modello (API Funzionale) ---
    # 1. Ingresso: sequenze di 10 numeri interi
    inputs = layers.Input(shape=(10,), dtype="int32", name="Ingresso_Indici")
    
    # 2. Il Layer Embedding (Il Traduttore Universale)
    # - input_dim: quante parole conosciamo.
    # - output_dim: quante coordinate ha ogni parola (50).
    # - embeddings_initializer: INIETTIAMO qui la nostra matrice ponte di GloVe.
    # - mask_zero=True: Dice alla rete di ignorare i '0' del padding (Slide 9).
    # - trainable=False: CONGELIAMO i pesi. Non vogliamo cambiare la saggezza di GloVe.
    embedding_layer = layers.Embedding(
        input_dim=num_tokens,
        output_dim=embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        mask_zero=True,
        trainable=False, 
        name="Memoria_GloVe"
    )
    
    # Colleghiamo i pezzi:
    x = embedding_layer(inputs)           # (Batch, 10) -> (Batch, 10, 50)
    x = layers.GlobalAveragePooling1D()(x) # (Batch, 10, 50) -> (Batch, 50) - Media della frase
    outputs = layers.Dense(1, activation="sigmoid")(x) # Output finale: probabilità 0-1
    
    # Compilazione
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ========================================================================================
# LOGICA DI ESECUZIONE (IL MAIN)
# ========================================================================================

# 1. Prepariamo i dati e il vettorizzatore
testi, labels, vectorizer = prepare_data()

# 2. Creiamo il cervello (Modello) caricando GloVe
# Notate come il modello 'incapsula' la conoscenza pre-addestrata
model = build_real_transfer_model(vectorizer)

# 3. Visione d'insieme del Modello
model.summary()

# --- ISPEZIONE FISICA (L'Anima della Parola) ---
# Vogliamo vedere il vettore numerico della parola 'fantastic'
print("\n--- ISPEZIONE LIVE (SLIDE 6) ---")

# Step 1: Trasformiamo 'fantastic' nel suo indice numerico
# Usiamo keras.ops per gestire correttamente la memoria se siamo su GPU
parola_test = ["fantastic"]
indices_tensore = vectorizer(parola_test)
indices_np = keras.ops.convert_to_numpy(indices_tensore)
idx = int(indices_np[0][0])

# Step 2: Estraiamo il vettore (i pesi) dal Layer Embedding per quell'indice
pesi_embedding = model.get_layer("Memoria_GloVe").get_weights()[0]
vettore = pesi_embedding[idx]

print(f"Parola: '{parola_test[0]}' -> Convertita in Indice: {idx}")
print(f"Prime 5 dimensioni del suo vettore GloVe reale:\n{vettore[:5]}...")

# 4. TEST DI INFERENZA FINALE
# Vediamo se il modello capisce il sentiment di una frase mai vista

# addestrimao il nostro modello
X_train = vectorizer(testi)
y_train = labels

print("\n[INFO] Inizio addestramento...")
model.fit(X_train, y_train, epochs=20, verbose=1)

# eseguiamo l'inferenza
nuova_frase = ["this movie is fantastic"]
test_input = vectorizer(nuova_frase)
predizione = model.predict(test_input, verbose=0)

# Estraiamo il valore scalare dalla predizione
probabilita = float(keras.ops.convert_to_numpy(predizione)[0][0])
sentiment = "POSITIVO" if probabilita > 0.5 else "NEGATIVO"

print(f"\n[RISULTATO]")
print(f"Frase: '{nuova_frase[0]}'")
print(f"Probabilità Positività: {probabilita:.4f} -> Sentiment: {sentiment}")