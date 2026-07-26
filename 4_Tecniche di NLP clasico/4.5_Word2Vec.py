import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np

# ======================================================================================
# 2. CARICAMENTO DELLO SPAZIO VETTORIALE (KNOWLEDGE RETRIEVAL)
# ======================================================================================
# Utilizziamo 'gensim' per scaricare un modello di Word Embedding pre-addestrato.
# Modello scelto: 'glove-twitter-25' (25 dimensioni, addestrato su Twitter).
# Logica: Le parole vengono trasformate in vettori numerici dove la vicinanza spaziale 
# indica una somiglianza semantica nel contesto del linguaggio naturale.
print("Download e caricamento dello spazio vettoriale in corso...")
word_vectors = api.load("glove-twitter-25") 

# ======================================================================================
# 3. INTERROGAZIONE SEMANTICA (ANALISI DELLE RELAZIONI)
# ======================================================================================
# Calcolo della similarità tramite Coseno (Cosine Similarity).
# Matematicamente: Misura l'angolo tra due vettori nel multi-spazio a 25 dimensioni.
# Più il valore è vicino a 1, più le parole sono state usate in contesti simili.
word_a, word_b = "pizza", "pasta"
sim = word_vectors.similarity(word_a, word_b)
print(f"\nSimilarità semantica tra '{word_a}' e '{word_b}': {sim:.4f}")

# ======================================================================================
# 4. ALGEBRA VETTORIALE (ANALOGIE SEMANTICHE)
# ======================================================================================
# Word2Vec permette operazioni matematiche sulle parole. 
# Equazione: Re - Uomo + Donna = Regina
# Spiegazione: Sottrarre il vettore 'man' da 'king' isola il concetto di 'regalità'. 
# Aggiungendo 'woman', ci spostiamo nel punto dello spazio che rappresenta la 'regalità femminile'.
print("\nRisoluzione analogia: Re - Uomo + Donna = ...")
result = word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Risultato calcolato: {result[0][0]} (Confidenza: {result[0][1]:.4f})")

# ======================================================================================
# 5. PONTE FRA GENSIM E KERAS (FEATURE TRANSFER)
# ======================================================================================
# Questa funzione è il "cuore" dell'integrazione: trasforma un dizionario di parole 
# (Gensim) in un layer neuronale (Keras) pronto per una rete profonda.

def create_keras_embedding(gensim_model):
    # Passaggio A: Recupero metadati dallo spazio Gensim
    # vocab_size: quante parole uniche conosce il modello (es. 1.2 milioni).
    # vector_dim: quanti numeri compongono ogni parola (qui 25).
    vocab_size = len(gensim_model.index_to_key)
    vector_dim = gensim_model.vector_size
    
    # Passaggio B: Estrazione della matrice dei pesi
    # 'weights' è un array NumPy di forma [vocab_size, vector_dim].
    # Ogni riga 'i' contiene il vettore della parola all'indice 'i' del vocabolario.
    weights = gensim_model.vectors
    
    # Passaggio C: Definizione del Layer Embedding
    # input_dim: Raggio d'azione del vocabolario (ID delle parole).
    # output_dim: Dimensione del vettore di uscita (25).
    # trainable=False: "Congeliamo" i pesi. Non vogliamo che la rete modifichi 
    # la conoscenza linguistica già appresa durante l'addestramento.
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_dim,
        trainable=False,
        name="word2vec_pretrained"
    )
    
    # Passaggio D: Inizializzazione fisica del Layer
    # keras.layers.build() alloca la memoria necessaria per i pesi.
    # set_weights() inietta i numeri presi da Gensim dentro il layer Keras.
    embedding_layer.build((None,))
    embedding_layer.set_weights([weights])
    
    return embedding_layer

# ======================================================================================
# 6. ARCHITETTURA DEL MODELLO (PIPELINE DI CLASSIFICAZIONE)
# ======================================================================================
# Costruiamo una rete che accetta testo (sotto forma di ID numerici) e produce un output.
model = keras.Sequential([
    # Input Layer: Accetta sequenze di lunghezza variabile (None) di interi.
    keras.layers.Input(shape=(None,), dtype="int32"),
    
    # Pretrained Embedding: Trasforma gli ID in vettori semantici da 25 dimensioni.
    create_keras_embedding(word_vectors),
    
    # GlobalAveragePooling1D: Se abbiamo una frase, calcola il vettore "medio" della frase.
    # Trasforma una matrice [numero_parole, 25] in un unico vettore [25].
    keras.layers.GlobalAveragePooling1D(), 
    
    # Dense Layer: Un singolo neurone con attivazione Sigmoide per classificazione binaria.
    keras.layers.Dense(1, activation="sigmoid")
])

# ======================================================================================
# 7. PERSISTENZA E SALVATAGGIO (FORMATO .KERAS)
# ======================================================================================
# Il nuovo formato .keras è un pacchetto compresso che contiene sia l'architettura 
# (come sono connessi i layer) sia i pesi pre-caricati da Gensim.
model_path = "word2vec_pipeline_2026.keras"
model.save(model_path)

print(f"\n[INFO] Architettura e Pesi integrati correttamente.")
print(f"[INFO] Modello salvato in: {model_path}")
print(f"[INFO] Dimensione vocabolario importato: {len(word_vectors.index_to_key)} parole.")