"""
================================================================================
MIGRAZIONE DA LSTM A GRU - BENCHMARK REALE (DATASET AG NEWS)
================================================================================
CONFRONTO: LSTM vs GRU
DATASET: AG News (Classificazione news: 4 classi)
BACKEND: Keras 3 (PyTorch)
================================================================================
"""

import os
import time
import numpy as np

# Impostiamo il backend Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers

# --- CARICAMENTO DATI (AG NEWS) ---
def load_ag_news(subset_size=10000):
    """
    Carica il dataset AG News tramite Hugging Face Datasets.
    Se non presente, lo installa al volo.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installazione di 'datasets' necessaria...")
        os.system("pip install datasets")
        from datasets import load_dataset

    print("Caricamento dataset AG News...")
    dataset = load_dataset("ag_news")
    
    # Prendiamo un subset per velocizzare il benchmark
    train_data = dataset["train"].shuffle(seed=42).select(range(subset_size))
    test_data = dataset["test"].shuffle(seed=42).select(range(int(subset_size * 0.2)))
    
    X_train = [item["text"] for item in train_data]
    y_train = [item["label"] for item in train_data]
    
    X_test = [item["text"] for item in test_data]
    y_test = [item["label"] for item in test_data]
    
    # One-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=4)
    y_test = keras.utils.to_categorical(y_test, num_classes=4)
    
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    return np.array(X_train), y_train, np.array(X_test), y_test, class_names

# --- ARCHITETTURA MODELLO ---

def build_benchmark_model(model_type, vocab_size, max_len, learning_rate=0.001):
    inputs = layers.Input(shape=(max_len,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=64)(inputs)
    
    if model_type == "LSTM":
        # LSTM: 64 unità, 4 gate -> più parametri
        rec_layer = layers.LSTM(64)(x)
    else:
        # GRU: 64 unità, 3 gate (reset, update, new memory) -> meno parametri
        rec_layer = layers.GRU(64)(x)
    
    x = layers.BatchNormalization()(rec_layer)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs, name=f"Model_{model_type}")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# --- ESECUZIONE ESPERIMENTO ---

def run_experiment():
    # 1. Preparazione Dati
    X_train_raw, y_train, X_test_raw, y_test, class_names = load_ag_news(subset_size=8000)
    
    max_features = 10000 
    max_len = 50 # Lunghezza media news
    
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_sequence_length=max_len
    )
    vectorize_layer.adapt(X_train_raw)
    
    X_train = vectorize_layer(X_train_raw)
    X_test = vectorize_layer(X_test_raw)
    
    results = {}

    # 2. Test LSTM
    print("\n" + "="*40)
    print("TEST 1: LSTM (Baseline)")
    print("="*40)
    model_lstm = build_benchmark_model("LSTM", max_features, max_len)
    model_lstm.summary()
    
    # Callback per salvare i migliori pesi sulla validazione
    checkpoint_lstm = keras.callbacks.ModelCheckpoint(
        filepath="best_lstm.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max"
    )
    
    epochs = 10
    start = time.time()
    history_lstm = model_lstm.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=64, 
        validation_split=0.1, 
        verbose=1,
        callbacks=[checkpoint_lstm]
    )
    duration_lstm = time.time() - start
    
    # Ripristiniamo i pesi migliori prima del test finale
    model_lstm.load_weights("best_lstm.weights.h5")
    
    acc_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)[1]
    results["LSTM"] = {"params": model_lstm.count_params(), "time": duration_lstm/epochs, "acc": acc_lstm}

    # 3. Test GRU
    print("\n" + "="*40)
    print("TEST 2: GRU (Migrazione)")
    print("="*40)
    model_gru = build_benchmark_model("GRU", max_features, max_len)
    model_gru.summary()
    
    # Callback per salvare i migliori pesi sulla validazione
    checkpoint_gru = keras.callbacks.ModelCheckpoint(
        filepath="best_gru.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max"
    )
    
    start = time.time()
    history_gru = model_gru.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=64, 
        validation_split=0.1, 
        verbose=1,
        callbacks=[checkpoint_gru]
    )
    duration_gru = time.time() - start
    
    # Ripristiniamo i pesi migliori prima del test finale
    model_gru.load_weights("best_gru.weights.h5")
    
    acc_gru = model_gru.evaluate(X_test, y_test, verbose=0)[1]
    results["GRU"] = {"params": model_gru.count_params(), "time": duration_gru/epochs, "acc": acc_gru}

    # 4. Challenge: GRU con LR ridotto
    print("\n" + "="*40)
    print("TEST 3: GRU (Learning Rate 0.0005)")
    print("="*40)
    model_gru_stabile = build_benchmark_model("GRU", max_features, max_len, learning_rate=0.0005)
    
    # Callback per salvare i migliori pesi sulla validazione
    checkpoint_stabile = keras.callbacks.ModelCheckpoint(
        filepath="best_gru_stabile.weights.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max"
    )
    
    history_gru_stabile = model_gru_stabile.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=64, 
        validation_split=0.1, 
        verbose=1,
        callbacks=[checkpoint_stabile]
    )
    
    # Ripristiniamo i pesi migliori prima del test finale
    model_gru_stabile.load_weights("best_gru_stabile.weights.h5")
    
    acc_gru_stabile = model_gru_stabile.evaluate(X_test, y_test, verbose=0)[1]

    # --- REPORT FINALE ---
    print("\n" + "#"*50)
    print("REPORT DI MIGRAZIONE: LSTM vs GRU")
    print("#"*50)
    print(f"{'Metrica':<25} | {'LSTM':<12} | {'GRU':<12}")
    print("-" * 55)
    print(f"{'Parametri Totali':<25} | {results['LSTM']['params']:<12} | {results['GRU']['params']:<12}")
    print(f"{'Tempo per Epoca (s)':<25} | {results['LSTM']['time']:<12.2f} | {results['GRU']['time']:<12.2f}")
    print(f"{'Accuratezza Finale':<25} | {results['LSTM']['acc']:<12.2%} | {results['GRU']['acc']:<12.2%}")
    print("-" * 55)
    print(f"Guadagno computazionale stimato: {((results['LSTM']['time'] - results['GRU']['time']) / results['LSTM']['time'])*100:.1f}%")
    print(f"Accuratezza GRU (LR ridotto): {acc_gru_stabile:.2%}")

if __name__ == "__main__":
    run_experiment()