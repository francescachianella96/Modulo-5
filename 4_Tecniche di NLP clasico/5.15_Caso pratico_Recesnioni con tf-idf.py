"""
================================================================================
Sentiment Analysis su Scala Reale con Keras 3 e PyTorch (Best Practices 2026)
================================================================================
Questo script mostra rappresenta una pipeline completa di Deep Learning per la 
Sentiment Analysis, partendo dal caricamento del dataset IMDb (50.000 recensioni)
fino alla creazione di una rete neurale densa utilizzando le ultime funzionalità
di Keras 3 con backend PyTorch.

Concetti chiave:
- Gestione Big Data e pulizia chirurgica del testo.
- Vettorizzazione TF-IDF ottimizzata per ridurre l'overfitting.
- Architettura Keras Funzionale per massima flessibilità.
- Strategie di regolarizzazione moderne (Dropout, Batch Normalization).
================================================================================
"""

import os

# Impostiamo il backend di Keras su PyTorch prima di importare Keras (Best Practice 2026)
os.environ["KERAS_BACKEND"] = "torch"
# Cruciale su Windows per evitare deadlock con i Transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import numpy as np
import pandas as pd
import torch
import keras
from keras import layers, models, ops
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from transformers import pipeline # Per il confronto con lo Stato dell'Arte
import kagglehub # Utilizzato per il download efficiente del dataset

def scarica_e_carica_dati():
    """
    Scarica il dataset IMDb da Kaggle e lo carica in un DataFrame Pandas.
    
    Returns:
        pd.DataFrame: DataFrame contenente le recensioni e i relativi sentiment.
    """
    print("[1/6] Scaricamento del dataset in corso...")
    # Scarichiamo il dataset ufficiale IMDb di 50k recensioni
    # Nota: Nel 2026 l'efficienza nel caricamento dati è prioritaria
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    csv_path = os.path.join(path, "IMDB Dataset.csv")
    
    # Carichiamo i dati assicurandoci dell'encoding UTF-8 (Slide 5)
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Mappiamo le etichette testuali in valori numerici (0 = Negativo, 1 = Positivo)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    return df

def pulizia_testo_chirurgica(text):
    """
    Esegue una pulizia profonda del testo rimuovendo tag HTML e rumore digitale.
    
    Args:
        text (str): Testo grezzo della recensione.
        
    Returns:
        str: Testo pulito e normalizzato.
    """
    # 1. Rimozione Tag HTML (Slide 5: <br />, ecc.) usando le espressioni regolari
    text = re.sub(r'<.*?>', ' ', text)
    
    # 2. Rimozione di caratteri non alfabetici e normalizzazione in minuscolo
    # Questo riduce il rumore e la dimensionalità del vocabolario (Slide 7)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    # 3. Rimozione di spazi bianchi superflui all'inizio e alla fine
    return text.strip()

def preprocessa_dataset(df):
    """
    Preprocessa l'intero dataset: pulizia, splitting e vettorizzazione TF-IDF.
    
    Args:
        df (pd.DataFrame): DataFrame originale.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
    """
    print("[2/6] Pulizia dei testi e Stratified Splitting...")
    # Applichiamo la pulizia a tutte le 50.000 recensioni
    df['review_cleaned'] = df['review'].apply(pulizia_testo_chirurgica)
    
    # Suddivisione Stratificata 80/20 (Slide 4 e 6)
    # Garantiamo che il bilanciamento 50/50 sia mantenuto in entrambi i set
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['review_cleaned'], 
        df['sentiment'], 
        test_size=0.20, 
        stratify=df['sentiment'], 
        random_state=42,
        shuffle=True # Fondamentale per evitare bias sull'ordine (Slide 5)
    )
    
    print("[3/6] Vettorizzazione TF-IDF con filtri statistici (Slide 8 & 9)...")
    # Configuriamo il TfidfVectorizer con le best practices discusse
    vectorizer = TfidfVectorizer(
        max_features=10000,   # Limitiamo a 10.000 termini per evitare 'maledizione dimensionalità'
        min_df=5,             # Scartiamo errori di battitura (parole che appaiono in < 5 doc)
        max_df=0.7,           # Scartiamo stop-words dinamiche (parole presenti in > 70% doc)
        ngram_range=(1, 2),    # Catturiamo il contesto locale (es: "non buono")
        sublinear_tf=True     # Applichiamo logaritmo alla frequenza (Slide 9)
    )
    
    # Trasformiamo i testi in matrici sparse
    X_train = vectorizer.fit_transform(X_train_raw).toarray()
    X_test = vectorizer.transform(X_test_raw).toarray()
    
    # In Keras 3 con backend PyTorch, le label per la classificazione binaria 
    # devono spesso essere in formato 2D (N, 1) per evitare errori nelle metriche
    y_train_2d = np.array(y_train).reshape(-1, 1)
    y_test_2d = np.array(y_test).reshape(-1, 1)
    
    return X_train, X_test, y_train_2d, y_test_2d, vectorizer, X_test_raw

def crea_modello_funzionale(input_dim):
    """
    Costruisce una rete neurale densa usando la Functional API di Keras 3.
    
    Args:
        input_dim (int): Numero di feature in ingresso (Max Features del TF-IDF).
        
    Returns:
        keras.Model: Il modello compilato.
    """
    print("[4/6] Costruzione del cervello digitale (Keras Functional API)...")
    
    # Definizione dell'Input Layer
    inputs = layers.Input(shape=(input_dim,), name="tfidf_input")
    
    # Primo blocco denso con Regolarizzazione (Slide 10)
    # Batch Normalization aiuta la stabilità del gradiente
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # Dropout al 50% per combattere l'overfitting
    
    # Secondo blocco denso più stretto (architettura a imbuto)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output Layer per classificazione binaria
    outputs = layers.Dense(1, activation='sigmoid', name="sentiment_output")(x)
    
    # Creazione del modello
    model = models.Model(inputs=inputs, outputs=outputs, name="IMDb_Sentiment_Analyzer")
    
    # Compilazione con ottimizzatore AdamW (standard industriale 2026)
    model.compile(
        optimizer="adamw",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.F1Score(average="micro", threshold=0.5)]
    )
    
    return model

def main():
    # 1. Caricamento Dati
    df = scarica_e_carica_dati()
    
    # 2. Preprocessing e Vettorizzazione
    X_train, X_test, y_train, y_test, vectorizer, X_test_raw = preprocessa_dataset(df)
    
    # 3. Creazione Modello
    input_dim = X_train.shape[1]
    model = crea_modello_funzionale(input_dim)
    model.summary()
    
    # 4. Addestramento con Early Stopping (Best Practice per evitare Overfitting)
    print("\n[5/6] Addestramento del modello...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2, # Usiamo una parte del training per la validazione interna
        epochs=20,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 5. Valutazione della Generalizzazione (Slide 11 & 12)
    print("\n[6/6] Valutazione della Generalizzazione (F1-Comparison)...")
    
    # Per un confronto equo e veloce (i Transformers sono pesanti su CPU), 
    # selezioniamo un campione di 500 recensioni dal Test Set
    sample_size = 500 
    X_test_sample_raw = X_test_raw.iloc[:sample_size]
    y_test_sample = y_test[:sample_size]
    
    # 1. Predizioni Modello Keras (Custom)
    X_test_sample_tfidf = vectorizer.transform(X_test_sample_raw).toarray()
    keras_preds_prob = model.predict(X_test_sample_tfidf, verbose=0)
    keras_preds = (keras_preds_prob > 0.5).astype(int).flatten()
    # Appiattiamo y_test_sample per passarlo a sklearn
    keras_f1 = f1_score(y_test_sample.flatten(), keras_preds)
    
    # 2. Predizioni Hugging Face (SOTA)
    print(f"Inizializzazione Hugging Face (SOTA) con framework PyTorch...")
    device = 0 if torch.cuda.is_available() else -1
    hf_pipeline = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        device=device, 
        framework="pt"
    )
    
    print(f"Calcolo predizioni Hugging Face su {sample_size} campioni...")
    hf_preds = []
    # Processiamo uno alla volta per massima stabilità su Windows (come nel file reference)
    for testo in tqdm(X_test_sample_raw.tolist(), desc="Analisi HF"):
        # Troncamento a 1000 caratteri per evitare sforamento 512 token
        res = hf_pipeline(testo[:1000], truncation=True)[0]
        hf_preds.append(1 if res['label'] == 'POSITIVE' else 0)
    
    hf_f1 = f1_score(y_test_sample.flatten(), hf_preds)
    
    print("-" * 50)
    print("RISULTATI CONFRONTO STATISTICO (F1-SCORE)")
    print("-" * 50)
    print(f"KERNAS CUSTOM (TF-IDF): {keras_f1:.4f}")
    print(f"HUGGING FACE (SOTA):    {hf_f1:.4f}")
    print("-" * 50)
    if hf_f1 > keras_f1:
        print(f"Risultato: Hugging Face vince di {(hf_f1 - keras_f1):.4f} punti!")
    else:
        print("Risultato: Il tuo modello custom tiene testa allo stato dell'arte!")
    
    # 7. Test Manuale Live e Confronto (Slide 14)
    test_frasi = [
        "This movie was an absolute masterpiece of modern cinema!",
        "The plot was boring and the acting was terrible.",
        "Not as good as the first one, but still worth a watch.",
        "Despite the rain, the film was a ray of sunshine." # Test sul contesto (Slide 14)
    ]
    
    print("\n--- SFIDA: Keras Personalizzato vs Hugging Face Pre-trained ---")
    for frase in test_frasi:
        # Analisi con Modello Keras (TF-IDF)
        frase_pulita = pulizia_testo_chirurgica(frase)
        vettore = vectorizer.transform([frase_pulita]).toarray()
        pred_keras = model.predict(vettore, verbose=0)[0][0]
        label_keras = "POSITIVO" if pred_keras > 0.5 else "NEGATIVO"
        conf_keras = pred_keras if pred_keras > 0.5 else 1 - pred_keras
        
        # Analisi con Hugging Face (Transformer)
        res_hf = hf_pipeline(frase)[0]
        label_hf = res_hf['label']
        conf_hf = res_hf['score']
        
        print(f"Frase: '{frase}'")
        print(f"  > KERAS (TF-IDF):   {label_keras} ({conf_keras:.2%})")
        print(f"  > HUGGING FACE:     {label_hf} ({conf_hf:.2%})")
        print("-" * 20)

if __name__ == "__main__":
    main()

# ==============================================================================
# SPIEGAZIONE DETTAGLIATA:
#
# 1. Impostazione Backend: 'os.environ["KERAS_BACKEND"] = "torch"' dice a Keras 
#    di usare PyTorch per il calcolo dei tensori invece di TensorFlow o JAX.
#
# 2. Pulizia: La Regex r'<.*?>' identifica tutto ciò che sta tra parentesi 
#    angolari (tag HTML) e lo sostituisce con uno spazio. Fondamentale per IMDb.
#
# 3. TF-IDF: Usiamo 'sublinear_tf=True' perché l'importanza di una parola non 
#    cresce in modo lineare con la sua frequenza (Slide 9). I 'Bigrams' permettono
#    di capire espressioni composte da due parole.
#
# 4. Keras Functional API: Rispetto al modello 'Sequential', l'API Funzionale 
#    specifica esplicitamente i collegamenti tra i layer (es. outputs=outputs(x)).
#    È lo standard richiesto per architetture complesse e multi-input.
#
# 5. Regolarizzazione: 'BatchNormalization' normalizza l'output di un layer per
#    velocizzare l'addestramento. 'Dropout' spegne casualmente dei neuroni per
#    costringere la rete a non memorizzare i dati (evitando l'Overfitting).
#
# 7. Hugging Face vs Keras: Mentre il TF-IDF osserva la statistica delle parole
#    nel nostro dataset specifico (veloce ma limitato), i Transformers di 
#    Hugging Face usano il 'Transfer Learning'. Hanno una comprensione 
#    profonda della grammatica e del sarcasmo, rendendoli spesso più 
#    precisi su frasi complesse ma più pesanti computazionalmente.
# ==============================================================================