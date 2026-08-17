"""
============================================================
ARCHITETTURA DI UNA RETE LSTM IN KERAS (BACKEND PYTORCH)
============================================================

Focus: Input -> Embedding -> LSTM -> Output.
"""

import os
# Impostiamo il backend a PyTorch (Best Practice Keras 3)
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers

def create_lstm_network(vocab_size=10000, max_len=200):
    """
    Definisce la struttura di una rete LSTM per Sentiment Analysis.
    
    Questa funzione utilizza la Functional API di Keras 
    per creare un grafo computazionale dove il testo fluisce 
    attraverso la memoria della LSTM.
    """
    
    # 1. DEFINIZIONE DELL'INPUT
    # Rappresenta una sequenza di numeri (indici di parole) di lunghezza fissa.
    inputs = keras.Input(shape=(max_len,), name="input_sequenza")

    # 2. LAYER EMBEDDING
    # Trasforma ogni numero in un vettore denso (es. 128 dimensioni).
    # È qui che il modello impara il "significato" iniziale delle parole.
    embedding = layers.Embedding(
        input_dim=vocab_size, 
        output_dim=128, 
        name="livello_embedding"
    )(inputs)

    # 3. LAYER LSTM (Il nucleo della lezione)
    # Parametri chiave:
    # - units=64: Dimensione dello stato nascosto e della 'Cell State'.
    # - dropout=0.2: Protezione contro l'overfitting (spegne neuroni a caso).
    # - return_sequences=False: Restituisce solo l'ultimo stato (il "riassunto" della frase).
    lstm_layer = layers.LSTM(
        units=64, 
        dropout=0.2, 
        recurrent_dropout=0.2, 
        name="memoria_lstm"
    )(embedding)

    # 4. LIVELLO DI OUTPUT
    # Un singolo neurone con attivazione Sigmoide.
    # Trasforma la memoria della LSTM in una probabilità (0=Negativo, 1=Positivo).
    outputs = layers.Dense(1, activation="sigmoid", name="classificatore")(lstm_layer)

    # 5. CREAZIONE DEL MODELLO
    # Specifichiamo dove inizia e dove finisce il flusso dei dati.
    model = keras.Model(inputs=inputs, outputs=outputs, name="Rete_LSTM_Semplice")

    # COMPILAZIONE
    # Configuriamo come il modello deve imparare.
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# --- VISUALIZZAZIONE ---

if __name__ == "__main__":
    # Creiamo l'istanza del modello
    my_lstm = create_lstm_network()

    # Mostriamo lo schema dell'architettura a video
    # Perfetto per mostrare agli studenti come cambiano le dimensioni dei dati (Shape)
    my_lstm.summary()

"""
============================================================
SPIEGAZIONE DETTAGLIATA DEL CODICE
============================================================
1. INPUT: Riceve la frase già trasformata in numeri (batch_size, 200).
2. EMBEDDING: Aggiunge una dimensione vettoriale (batch_size, 200, 128).
3. LSTM: È qui che avviene il "Gating". Il layer elabora i 200 step temporali 
   uno dopo l'altro, ma grazie alla Cell State non dimentica l'inizio.
   L'output viene ridotto a (batch_size, 64).
4. DENSE: Prende i 64 concetti estratti dalla LSTM e decide il sentiment finale.
============================================================
"""