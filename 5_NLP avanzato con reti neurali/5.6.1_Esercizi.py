import os

# =================================================================
# 1. SETUP DELL'AMBIENTE
# =================================================================
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

# --- CONFIGURAZIONI GLOBALI ---
VOCABOLARIO_SIZE = 10000
LUNGHEZZA_MAX = 150
CATEGORIE = 46
EPOCHS = 10
BATCH_SIZE = 128

# =================================================================
# 2. PREPARAZIONE DATI
# =================================================================
def prepara_dataset():
    print("[1] Caricamento Dataset Reuters...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=VOCABOLARIO_SIZE)
    
    x_train = keras.utils.pad_sequences(x_train, maxlen=LUNGHEZZA_MAX)
    x_test = keras.utils.pad_sequences(x_test, maxlen=LUNGHEZZA_MAX)
    
    y_train = keras.utils.to_categorical(y_train, CATEGORIE)
    y_test = keras.utils.to_categorical(y_test, CATEGORIE)
    
    return (x_train, y_train), (x_test, y_test)

# =================================================================
# 3. COSTRUZIONE MODELLI
# =================================================================

def build_model(mode="uni", merge_mode=None):
    """
    Crea un modello basato sulla configurazione richiesta.
    mode: 'uni' per Unidirezionale, 'bi' per Bidirezionale.
    merge_mode: 'sum', 'concat', etc. (solo per 'bi').
    """
    inputs = keras.Input(shape=(LUNGHEZZA_MAX,))
    x = layers.Embedding(VOCABOLARIO_SIZE, 128)(inputs)
    
    if mode == "bi":
        # LSTM a 32 unità bidirezionale
        x = layers.Bidirectional(layers.LSTM(32), merge_mode=merge_mode)(x)
    else:
        # LSTM a 32 unità unidirezionale
        x = layers.LSTM(32)(x)
    
    outputs = layers.Dense(CATEGORIE, activation="softmax")(x)
    
    name = f"Modello_{mode}"
    if merge_mode: name += f"_{merge_mode}"
    
    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer="adamw", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# =================================================================
# 4. ESECUZIONE SFIDA
# =================================================================
def main():
    (x_train, y_train), (x_test, y_test) = prepara_dataset()
    
    # --- ANALISI PARAMETRICI ---
    print("\n" + "="*50)
    print("ANALISI ARCHITETTURALE")
    print("="*50)
    
    model_sum = build_model(mode="bi", merge_mode="sum")
    model_concat = build_model(mode="bi", merge_mode="concat") # Default
    
    print("\n--- MODELLO BI-LSTM (SUM) ---")
    model_sum.summary()
    
    print("\n--- MODELLO BI-LSTM (CONCAT) ---")
    model_concat.summary()
    
    # NOTA: Il numero di parametri dello strato Bidirectional è IDENTICO.
    # Tuttavia, lo strato Dense successivo ha meno parametri in 'sum' perché 
    # riceve 32 input invece di 64 (32+32).
    
    # --- ADDESTRAMENTO E CONFRONTO ---
    print(f"ADDESTRAMENTO ({EPOCHS} EPOCHE)")
    print("="*50)
    
    # Modello Bidirezionale (SUM)
    print("\nTraining Bi-LSTM (merge_mode='sum')...")
    history_bi = model_sum.fit(
        x_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1, 
        verbose=1
    )
    acc_bi = model_sum.evaluate(x_test, y_test, verbose=0)[1]
    
    # Modello Unidirezionale
    model_uni = build_model(mode="uni")
    print("\nTraining Unidirectional LSTM...")
    history_uni = model_uni.fit(
        x_train, y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_split=0.1, 
        verbose=1
    )
    acc_uni = model_uni.evaluate(x_test, y_test, verbose=0)[1]
    
    # --- RISULTATI FINALI ---
    print("\n" + "="*50)
    print("CONFRONTO FINALE")
    print("="*50)
    print(f"Accuratezza Bi-LSTM (Sum): {acc_bi:.2%}")
    print(f"Accuratezza Uni-LSTM:      {acc_uni:.2%}")
    
    improvement = ((acc_bi - acc_uni) / acc_uni) * 100
    print(f"\nIl modello Bidirezionale ha un incremento di performance del {improvement:+.2f}% rispetto all'unidirezionale.")
    
    # Verifica dello spazio di output (usando la proprietà output del layer che è corretta in Keras 3)
    try:
        output_dim = model_sum.layers[2].output.shape[-1]
        if output_dim == 32:
            print("\nVerifica Output Space: Corretto. 'sum' mantiene 32 unità in uscita,")
            print("consentendo di usare la stessa dimensione dello strato LSTM originale.")
    except Exception:
        print("\nNota: Impossibile verificare dinamicamente lo shape, controllare model.summary().")

if __name__ == "__main__":
    main()