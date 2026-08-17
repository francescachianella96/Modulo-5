import os

# --- CONFIGURAZIONE BACKEND ---
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

def crea_dataset_esteso():
    """
    FASE 1: DATASET DI 10 FRASI.
    Includiamo frasi di diversa natura per testare la capacità di memoria.
    """
    testi = [
        "Il gatto rincorre il topo velocemente",
        "Oggi il cielo è limpido e azzurro sopra le montagne",
        "Il Deep Learning rivoluziona il modo di elaborare dati",
        "La memoria delle reti neurali è complessa e affascinante",
        "L'intelligenza artificiale sta cambiando il nostro futuro",
        "Le sequenze temporali richiedono modelli con memoria",
        "Le reti ricorrenti semplici hanno limiti nel lungo termine",
        "I gated recurrent units migliorano la gestione del gradiente",
        "Il pre-padding aiuta a mantenere l'informazione finale",
        "Analisi avanzata del linguaggio naturale con Keras 3"
    ]
    # Etichette simulate (es. 1 per frasi tecniche, 0 per frasi comuni - puramente didattico)
    etichette = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype="float32")
    return testi, etichette

def preprocessamento_con_pre_padding(testi):
    """
    FASE 2: VETTORIZZAZIONE E PRE-PADDING.
    Spostiamo i token nulli (0) all'inizio della sequenza.
    """
    # 1. Creiamo il vectorizer senza definire output_sequence_length per gestirlo manualmente
    vectorizer = layers.TextVectorization(max_tokens=100)
    vectorizer.adapt(testi)
    
    # 2. Generiamo le sequenze (di lunghezza variabile inizialmente)
    sequenze_ragged = vectorizer(testi)
    
    # 3. Applichiamo PRE-PADDING: aggiungiamo zeri ALL'INIZIO invece che alla fine
    # Usiamo keras.ops.convert_to_numpy per gestire correttamente i tensor su GPU (CUDA)
    sequenze_np = keras.ops.convert_to_numpy(sequenze_ragged)
    
    sequenze_pre_padded = keras.utils.pad_sequences(
        sequenze_np, 
        maxlen=15, 
        padding='pre'
    )
    
    return sequenze_pre_padded, vectorizer

def build_comparative_model(vocab_size, model_type="SimpleRNN"):
    """
    FASE 3: COSTRUZIONE MODELLI PER CONFRONTO.
    """
    inputs = keras.Input(shape=(15,), name="Input_Tokens")
    
    # Embedding: traduce numeri in vettori densi
    x = layers.Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True)(inputs)
    
    # Selezione del layer ricorrente
    if model_type == "SimpleRNN":
        # SimpleRNN: Sente molto il peso del "tempo", tende a dimenticare l'inizio
        x = layers.SimpleRNN(32, name="Memoria_SimpleRNN")(x)
    elif model_type == "GRU":
        # GRU (Gated Recurrent Unit): Usa i gate per decidere cosa mantenere
        x = layers.GRU(32, name="Memoria_GRU")(x)
    
    outputs = layers.Dense(1, activation="sigmoid", name="Output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Modello_{model_type}")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- ESECUZIONE ---

# 1. Caricamento dati
testi_raw, y_train = crea_dataset_esteso()

# 2. Pre-processamento con Pre-Padding
x_train, vect_layer = preprocessamento_con_pre_padding(testi_raw)
vocab_size = len(vect_layer.get_vocabulary())

print(f"\n[INFO] Shape dei dati con Pre-Padding: {x_train.shape}")
print(f"[INFO] Esempio frase 1 (Pre-Padded):\n{x_train[0]}")
print("(Gli zeri sono all'inizio, le parole informative alla fine)")

# 3. Creazione dei due modelli
model_rnn = build_comparative_model(vocab_size, "SimpleRNN")
model_gru = build_comparative_model(vocab_size, "GRU")

# 4. Sintesi dei modelli
print("\n--- ARCHITETTURA SIMPLERNN ---")
model_rnn.summary()

print("\n--- ARCHITETTURA GRU ---")
model_gru.summary()

# --- ANALISI E CONCLUSIONI ---
print("\n" + "="*60)
print("ANALISI CRITICA: PRE-PADDING E ARCHITETTURE")
print("="*60)
print("""
1. IMPATTO DEL PRE-PADDING:
   Spostando i token nulli all'INIZIO della sequenza, le parole reali arrivano 
   per ultime al layer ricorrente. Questo è un vantaggio enorme per SimpleRNN:
   essendo l'ultimo input elaborato quello più vicino all'output, lo stato 
   finale della memoria non viene 'diluito' dall'elaborazione di zeri finali.
   Se avessimo usato il Post-Padding, SimpleRNN avrebbe 'masticato' molti zeri 
   DOPO aver letto la frase, rischiando di dimenticare il contenuto utile.

2. CONFRONTO SIMPLE-RNN VS GRU:
   - SimpleRNN: Anche con il pre-padding, soffre del problema del gradiente 
     che svanisce se la frase è molto lunga. Tende a dare importanza solo 
     alle ultimissime parole lette.
   - GRU: Grazie ai suoi 'Update Gate' e 'Reset Gate', è in grado di decidere 
     se una parola letta all'inizio è ancora importante alla fine. Mantiene 
     il contesto iniziale sensibilmente MEGLIO rispetto a una SimpleRNN, 
     poiché può bypassare selettivamente i passaggi temporali.

CONCLUSIONE: 
La GRU è l'architettura superiore per mantenere il contesto, ma il PRE-PADDING 
è una best practice fondamentale per entrambi i modelli per evitare che la 
fase di padding 'inquini' lo stato finale della memoria.
""")