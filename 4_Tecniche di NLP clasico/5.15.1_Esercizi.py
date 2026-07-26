"""
================================================================================
Sentiment Analysis su Scala Reale con Keras 3 e PyTorch (Aggiornato)
================================================================================
Modifiche apportate:
- TfidfVectorizer: max_features aumentato a 20.000.
- TfidfVectorizer: ngram_range esteso a (1, 3) per includere i trigrammi.
- Aggiunta funzione 'predici_sentiment_live' per input di testo grezzo.
================================================================================
"""

import os

# Impostiamo il backend di Keras su PyTorch prima di importare Keras
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import numpy as np
import pandas as pd
import torch
import keras
from keras import layers, models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from transformers import pipeline
import kagglehub

def scarica_e_carica_dati():
    """Scarica il dataset IMDb da Kaggle."""
    print("[1/6] Scaricamento del dataset in corso...")
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_path = os.path.join(path, "IMDB Dataset.csv")
    df = pd.read_csv(csv_path, encoding='utf-8')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

def pulizia_testo_chirurgica(text):
    """Esegue la pulizia profonda del testo."""
    text = re.sub(r'<.*?>', ' ', text) # Rimozione HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower() # Solo lettere e minuscolo
    return text.strip()

def preprocessa_dataset(df):
    """Preprocessa il dataset con TF-IDF a 20k feature e trigrammi."""
    print("[2/6] Pulizia dei testi e Stratified Splitting...")
    df['review_cleaned'] = df['review'].apply(pulizia_testo_chirurgica)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['review_cleaned'], 
        df['sentiment'], 
        test_size=0.20, 
        stratify=df['sentiment'], 
        random_state=42,
        shuffle=True
    )
    
    print("[3/6] Vettorizzazione TF-IDF (20.000 feature, Trigrammi)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,   # Modificato da 10k a 20k (Richiesta Utente)
        min_df=5,             
        max_df=0.7,           
        ngram_range=(1, 3),    # Modificato da (1, 2) a (1, 3) (Richiesta Utente)
        sublinear_tf=True     
    )
    
    X_train = vectorizer.fit_transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()
    
    y_train_2d = np.array(y_train).reshape(-1, 1)
    y_test_2d = np.array(y_test).reshape(-1, 1)
    
    return X_train, X_test, y_train_2d, y_test_2d, vectorizer, X_test_raw

def crea_modello_funzionale(input_dim):
    """Costruisce la rete neurale densa."""
    print("[4/6] Costruzione del modello...")
    inputs = layers.Input(shape=(input_dim,), name="tfidf_input")
    
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) 
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name="sentiment_output")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="IMDb_Sentiment_Analyzer")
    model.compile(
        optimizer="adamw",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- NUOVA FUNZIONE RICHIESTA ---
def predici_sentiment_live(testo_grezzo, vectorizer, model):
    """
    Accetta testo grezzo, lo pulisce e restituisce 'Positivo' o 'Negativo'.
    """
    # 1. Pulizia
    testo_pulito = pulizia_testo_chirurgica(testo_grezzo)
    # 2. Trasformazione in vettore (TF-IDF)
    vettore_tfidf = vectorizer.transform([testo_pulito]).toarray()
    # 3. Predizione
    probabilita = model.predict(vettore_tfidf, verbose=0)[0][0]
    
    return "Positivo" if probabilita > 0.5 else "Negativo", probabilita

def main():
    # 1. Caricamento Dati
    df = scarica_e_carica_dati()
    
    # 2. Preprocessing
    X_train, X_test, y_train, y_test, vectorizer, X_test_raw = preprocessa_dataset(df)
    
    # 3. Creazione Modello
    input_dim = X_train.shape[1]
    model = crea_modello_funzionale(input_dim)
    
    # 4. Addestramento
    print("\n[5/6] Addestramento del modello...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15, # Spesso con più feature bastano meno epoche
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 5. Valutazione e Test della nuova funzione
    print("\n[6/6] Test della funzione di predizione live...")
    
    test_frasi = [
        "This movie was not so good, a waste of time.", # Trigramma "not so good"
        "Absolutely a masterpiece! I loved every single second.",
        "The plot was okay, but the acting could be better."
    ]
    
    print("\n--- RISULTATI PREDIZIONE LIVE ---")
    for frase in test_frasi:
        sent, score = predici_sentiment_live(frase, vectorizer, model)
        print(f"Testo: '{frase}'")
        print(f"Predizione: {sent} (Score: {score:.4f})")
        print("-" * 30)

if __name__ == "__main__":
    main()