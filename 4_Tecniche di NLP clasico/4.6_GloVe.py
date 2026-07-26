# ================================================================================
# Implementazione  di GloVe (Global Vectors)
# ================================================================================
# Questo script è una simulazione completa del flusso di lavoro di GloVe, dall'analisi 
# del testo "sporco" alla creazione di uno spazio vettoriale dove le parole simili 
# sono vicine tra loro.

# ARCHITETTURA DEL CODICE:
# 1. [ELABORAZIONE DATI]: Scarichiamo testi reali e costruiamo la "Mappa dei Vicini" 
#    (Matrice di Co-occorrenza). È qui che applichiamo la statistica globale.
# 2. [IL MODELLO]: Creiamo una rete neurale "minima" che ha un solo compito: 
#    regolare i vettori affinché il loro prodotto scalare rispecchi i conteggi fatti.
# 3. [ANALISI]: Interroghiamo la mappa creata per vedere se le parole "spazio" 
#    e "orbita" sono effettivamente finite vicine.

# REQUISITI: scikit-learn, keras, torch, numpy
# """

import os

# --- STEP 0: CONFIGURAZIONE INFRASTRUTTURA ---
# Impostiamo PyTorch come motore di calcolo prima di caricare Keras.
# Questo garantisce che tutte le operazioni 'ops' usino i tensori di PyTorch.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import layers, ops
from sklearn.datasets import fetch_20newsgroups # Per scaricare testi reali
import re # Per la pulizia del testo (Regex)
from collections import Counter, defaultdict # Per contare le parole in modo efficiente


# ==============================================================================
# FASE 1: COSTRUZIONE DEL CERVELLO STATISTICO (LA MATRICE DI CO-OCCORRENZA)
# ==============================================================================

def prepare_real_cooccurrence(vocab_size=1000, window_size=5):
    """
    # Legge il testo e crea la struttura di dati richiesta da GloVe.
    
    # Interazione: Questa funzione 'alimenta' il modello Keras trasformando 
    # frasi umane in triplette numeriche: (Indice_Parola_1, Indice_Parola_2, Forza_Legame).
    # """
    
    # 1.1 ACQUISIZIONE DATI
    print("1. Scaricamento testi reali (Articoli scientifici su spazio e medicina)...")
    data = fetch_20newsgroups(subset='train', categories=['sci.space', 'sci.med'])
    texts = data.data[:500] # Limite a 500 per non rallentare la lezione

    # 1.2 PULIZIA E TOKENIZZAZIONE (Trasformiamo "Ciao!" in ["ciao"])
    all_tokens = []
    word_pool = []
    for text in texts:
        # re.findall(r'\b[a-z]{3,}\b') -> Estrae solo parole di almeno 3 lettere, ignorando numeri e punteggiatura
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        all_tokens.append(tokens)
        word_pool.extend(tokens) # Mettiamo tutte le parole in un unico 'secchio' per contarle

    # 1.3 CREAZIONE VOCABOLARIO (Mappa: Parola -> Numero)
    # Prendiamo le 1000 parole più comuni per evitare di gestire milioni di termini rari
    most_common = Counter(word_pool).most_common(vocab_size)
    word_to_id = {word: i for i, (word, count) in enumerate(most_common)}
    id_to_word = {i: word for word, i in word_to_id.items()} # Inversa: serve per leggere i risultati

    # 1.4 IL CUORE DI GLOVE: SCANSIONE DEI VICINI (Slide 7-8)
    print("2. Scansione testo: Analisi della 'Sliding Window' (Chi sta vicino a chi?)...")
    cooc_matrix = defaultdict(float)

    for tokens in all_tokens:
        for i, word in enumerate(tokens):
            if word not in word_to_id: continue # Saltiamo parole fuori dal top-1000
            
            # Definiamo i confini della finestra (context window)
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            
            for j in range(start, end):
                if i == j: continue # Una parola non è vicina a se stessa
                neighbor = tokens[j]
                if neighbor not in word_to_id: continue
                
                # Slide 8: Pesatura dei vicini. Le parole adiacenti (distanza 1) 
                # contano 1.0, quelle più lontane (distanza 2) contano 0.5.
                dist = abs(i - j)
                cooc_matrix[(word_to_id[word], word_to_id[neighbor])] += 1.0 / dist

    # 1.5 FORMATTAZIONE PER IL MODELLO
    # Trasformiamo il dizionario in vettori NumPy pronti per l'addestramento
    idx_a, idx_b, counts = [], [], []
    for (a, b), weight in cooc_matrix.items():
        idx_a.append(a)
        idx_b.append(b)
        counts.append(weight)

    idx_a = np.array(idx_a, dtype="int32")
    idx_b = np.array(idx_b, dtype="int32")
    counts = np.array(counts, dtype="float32")

    # TRASFORMAZIONE LOGARITMICA (Slide 4)
    # GloVe non cerca di indovinare il conteggio esatto, ma il suo LOGARITMO.
    # Questo perché i conteggi crescono in modo esponenziale (legge di Zipf).
    log_targets = np.log(counts + 1e-10) # + 1e-10 evita il log(0) che è errore

    # FUNZIONE DI PESO (Slide 9-10)
    # Diciamo al modello: "Impegnati molto se vedi conteggi alti, ma ignora il rumore di fondo"
    sample_weights = np.minimum(1.0, (counts / 100.0)**0.75)

    return (idx_a, idx_b), log_targets, sample_weights, id_to_word


# ==============================================================================
# FASE 2: COSTRUZIONE DELLA RETE NEURALE (IL MOTORE GEOMETRICO)
# ==============================================================================

def build_glove_model(vocab_size, dimension=50):
    """
    Crea il modello Keras seguendo la logica Log-Bilinear.
    
    Interazione: Il modello prende due numeri (ID parole) e cerca di 
    far sì che i loro vettori interni, se moltiplicati tra loro, restituiscano 
    il logaritmo del conteggio che abbiamo calcolato prima.
    """
    
    # Input: Rappresentano un indice di parola (es. "space" è l'indice 45)
    input_target = layers.Input(shape=(1,), name="Parola_A")
    input_context = layers.Input(shape=(1,), name="Parola_B")
    
    # LAYER EMBEDDING: È il cuore del Deep Learning per NLP. 
    # È una tabella gigante dove ogni parola ha una riga di 'dimension' numeri.
    # Questi numeri all'inizio sono casuali, alla fine saranno 'significato'.
    embedding_layer = layers.Embedding(vocab_size, dimension, name="Mappa_Vettoriale")
    
    # Estraiamo i vettori per le due parole in input
    vec_a = embedding_layer(input_target)
    vec_b = embedding_layer(input_context)
    
    # PRODOTTO SCALARE (Dot Product): Misura la sovrapposizione tra i due vettori.
    # Slide 4: L'obiettivo è dot(vec_a, vec_b) = log(conteggio_reale)
    dot_product = layers.Dot(axes=2)([vec_a, vec_b])
    
    # Puliamo la forma dei dati (da [Batch, 1, 1] a [Batch, 1])
    output = layers.Reshape((1,))(dot_product)
    
    # Configurazione finale: Usiamo Adam (ottimizzatore intelligente) e MSE (errore quadratico)
    model = keras.Model(inputs=[input_target, input_context], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")
    
    return model


# ==============================================================================
# FASE 3: TRAINING E INTERROGAZIONE (VERIFICA DEI RISULTATI)
# ==============================================================================

# 3.1 Prepariamo tutto
(inputs, targets, weights, id_to_word) = prepare_real_cooccurrence()
model = build_glove_model(vocab_size=len(id_to_word))

# 3.2 ADDESTRAMENTO
print("\n3. Inizio fase di apprendimento... (Il modello sposta i vettori nello spazio)")
# Passiamo sia la coppia di parole che i pesi (weights) per dare importanza ai dati giusti
model.fit(
    x=[inputs[0], inputs[1]], 
    y=targets, 
    sample_weight=weights, 
    epochs=12, 
    batch_size=512, 
    verbose=1
)

# 3.3 ESTRAZIONE CONOSCENZA
# Ora che il modello ha imparato, prendiamo i vettori definitivi
final_vectors = model.get_layer("Mappa_Vettoriale").get_weights()[0]

def find_similar(word_to_test, n=5):
    """Calcola la similarità tra una parola e tutto il resto del vocabolario."""
    # Cerchiamo l'indice della parola (se esiste)
    word_id = None
    for i, w in id_to_word.items():
        if w == word_to_test:
            word_id = i
            break
    
    if word_id is None: return f"'{word_to_test}' non trovata."

    # Prendiamo il suo vettore
    target_v = final_vectors[word_id]
    
    # SIMILARITÀ DEL COSENO (Slide 12)
    # È la formula standard per vedere se due frecce (vettori) puntano nella stessa direzione:
    # cos(theta) = (v1 dot v2) / (||v1|| * ||v2||)
    norms = np.linalg.norm(final_vectors, axis=1) # Lunghezza di ogni vettore nel vocabolario
    target_norm = np.linalg.norm(target_v)        # Lunghezza del nostro vettore test
    
    # Calcoliamo la vicinanza con TUTTI gli altri vettori contemporaneamente (efficienza NumPy)
    similarities = np.dot(final_vectors, target_v) / (norms * target_norm + 1e-10)
    
    # Ordiniamo e prendiamo i migliori (saltando il primo perché è la parola stessa)
    best_indices = np.argsort(similarities)[::-1][1:n+1]
    
    return [(id_to_word[idx], similarities[idx]) for idx in best_indices]

# TEST FINALE: Vediamo se il modello ha capito la semantica del dataset
print("\n--- ANALISI SEMANTICA GLOBALE ---")
for word in ["space", "health", "orbit", "system"]:
    results = find_similar(word)
    print(f"Parole simili a '{word}': {results}")


# ==============================================================================
# CONCLUSIONI DIDATTICHE
# ==============================================================================
"""
COSA ABBIAMO DIMOSTRATO:
1. GloVe non "legge" il testo durante l'allenamento: legge solo la matrice pre-calcolata. 
#    Questo lo rende incredibilmente veloce su grandi mole di dati.
# 2. La funzione di pesatura garantisce che parole comuni (come 'the', ignora qui 
#    ma presente in testi grossi) non schiaccino i termini tecnici come 'orbit'.
# 3. Quello che vedi come 'Similarità' è la prova che la statistica si è trasformata 
#    in geometria: parole che co-occorrono spesso nell'intero corpus finiscono per 
#    avere vettori simili nello spazio multidimensionale.
# """