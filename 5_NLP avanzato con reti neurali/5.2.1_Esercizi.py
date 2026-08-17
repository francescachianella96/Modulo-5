import os

# FASE 0: SETUP DEL MOTORE DI CALCOLO
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

def risolvi_esercizio():
    # 1. Set di 3 frasi di esempio
    frasi_esempio = [
        "Il deep learning è affascinante",
        "Keras rende tutto più semplice",
        "L'elaborazione del linguaggio naturale evolve rapidamente"
    ]
    
    print(f"--- FASE 1: PREPARAZIONE DATI ---")
    # 2. TextVectorization con vocabolario di 5000 parole e padding a 15
    vectorizer = layers.TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=15 
    )
    
    # Adattiamo il vettorizzatore sulle frasi
    vectorizer.adapt(frasi_esempio)
    
    # Applichiamo il padding alle frasi
    vettori_input = vectorizer(frasi_esempio)
    print(f"Frasi vettorizzate (con padding a 15):\n{vettori_input}\n")
    
    # 3. Costruzione del Modello
    def create_model(trainable=False):
        inputs = layers.Input(shape=(15,), dtype="int32")
        
        # Layer di Embedding: 5000 parole, 100 dimensioni, masking supportato
        embedding_layer = layers.Embedding(
            input_dim=5000,
            output_dim=100,
            mask_zero=True,
            trainable=trainable,
            name="embedding_layer"
        )
        
        x = embedding_layer(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    # --- CASO 1: trainable=False (Statico) ---
    print("--- CASO 1: Pesi NON addestrabili (trainable=False) ---")
    model_static = create_model(trainable=False)
    model_static.summary()
    
    # --- CASO 2: trainable=True (Dinamico) ---
    print("\n--- CASO 2: Pesi addestrabili (trainable=True) ---")
    model_trainable = create_model(trainable=True)
    model_trainable.summary()
    
    # Analizziamo la differenza
    params_static = model_static.count_params()
    trainable_params_static = sum(np.prod(v.shape) for v in model_static.trainable_weights)
    
    params_trainable = model_trainable.count_params()
    trainable_params_dynamic = sum(np.prod(v.shape) for v in model_trainable.trainable_weights)
    
    print("\n--- ANALISI SFIDA ---")
    print(f"Parametri Totali (Statico): {params_static}")
    print(f"Parametri Addestrabili (Statico): {trainable_params_static}")
    print(f"Parametri Totali (Dinamico): {params_trainable}")
    print(f"Parametri Addestrabili (Dinamico): {trainable_params_dynamic}")
    print(f"Differenza parametri addestrabili: {trainable_params_dynamic - trainable_params_static}")

if __name__ == "__main__":
    risolvi_esercizio()