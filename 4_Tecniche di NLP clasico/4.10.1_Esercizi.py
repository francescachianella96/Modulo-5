"""
================================================================================
PIPELINE DI SPAM DETECTION
================================================================================
Questo script rappresenta una "Pipeline" per la classificazione di SMS,
ottimizzata per identificare messaggi camuffati e gestire lo slang tipico.

MODIFICHE RISPETTO ALLA VERSIONE BASE:
1. REGEX AVANZATA: Identifica e corregge parole con spazi (es: "F r e e" -> "Free").
2. TEST CRITICO: Verifica la corretta traduzione dello slang "u" -> "you".
3. ANALISI BILANCIAMENTO: Confronto tra Naive Bayes e SVM sulla sensibilità ai dati sbilanciati.

"""

import os
import re
import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from nltk.stem import PorterStemmer

# CONFIGURAZIONE BACKEND
os.environ["KERAS_BACKEND"] = "torch"

# --- 1. COMPONENTE: GESTIONE DELLA CONOSCENZA ---

def get_slang_map():
    """Ritorna una mappa di traduzione per il linguaggio SMS."""
    return {
        "u": "you",
        "r": "are",
        "2": "to",
        "4": "for",
        "txt": "text",
        "msg": "message",
        "c": "see",
        "b": "be"
    }

# --- 2. COMPONENTE: PREPROCESSING  ---

def preprocess_sms(text, stemmer=PorterStemmer()):
    """
    Esegue la "pulizia profonda" con correzione di camuffamento e slang.
    """
    # 1. Normalizzazione
    text = text.lower()
    
    # 2. [NOVITÀ] Correzione parole camuffate (es: "F r e e" -> "Free")
    # Questa Regex cerca una singola lettera seguita da uno spazio,
    # purché anche la lettera successiva sia singola.
    text = re.sub(r'(?i)(?<=\b[a-z])\s+(?=[a-z]\b)', '', text)
    
    # 3. Espansione Slang
    slang_map = get_slang_map()
    words = text.split()
    text = " ".join([slang_map.get(w, w) for w in words])
    
    # 4. Tokenizzazione URL e Numeri
    text = re.sub(r'http[s]?://\S+', 'url_token', text)
    text = re.sub(r'\d+', 'num_token', text)
    
    # 5. Filtro Caratteri
    text = re.sub(r'[^a-zA-Z!\s]', '', text)
    
    # 6. Stemming
    text = " ".join([stemmer.stem(w) for w in text.split()])
    
    return text

# --- 3. COMPONENTE: MODELLO NEURALE ---

def build_keras_classifier(input_shape):
    inputs = keras.Input(shape=(input_shape,), name="tfidf_input")
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SpamDetector_Keras")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- 4. MOTORE DI ESECUZIONE ---

def main():
    # A. CARICAMENTO DATASET
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    print(f"--- Caricamento Dataset ---")
    try:
        df = pd.read_csv(url, sep='\t', names=['label_raw', 'text'])
    except:
        df = pd.DataFrame({'text': ["Free money", "Hi there"], 'label_raw': ['spam', 'ham']})

    # Aggiunta messaggio critico con slang "u"
    critical_msg = "u should come to the party" # Dovrebbe essere Ham (0)
    new_row = pd.DataFrame({'label_raw': ['ham'], 'text': [critical_msg]})
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Test camuffamento (opzionale: lo aggiungiamo per vedere se funziona)
    df = pd.concat([df, pd.DataFrame({'label_raw':['spam'], 'text':['F r e e  m o n e y']})], ignore_index=True)

    df['label'] = df['label_raw'].map({'ham': 0, 'spam': 1})

    # B. PREPROCESSING
    print("--- Preprocessing in corso... ---")
    df['clean_text'] = df['text'].apply(preprocess_sms)

    # C. TRASFORMAZIONE TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label'].values

    # Identifichiamo l'indice del messaggio critico per verificarlo dopo
    critical_idx = len(df) - 2 # Era il penultimo aggiunto

    # D. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # E. ALLENAMENTO E CONFRONTO
    print("\n--- Allenamento Modelli ---")
    
    # 1. Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    # 2. SVM Lineare
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)

    # F. VERIFICA MESSAGGIO CRITICO
    # Vediamo come viene classificato il messaggio "u should come to the party"
    critical_vec = vectorizer.transform([preprocess_sms(critical_msg)]).toarray()
    nb_crit = nb_model.predict(critical_vec)[0]
    svm_crit = svm_model.predict(critical_vec)[0]
    
    print(f"\nVerifica Messaggio Critico: '{critical_msg}'")
    print(f"Post-preprocessing: '{preprocess_sms(critical_msg)}'")
    print(f"Predizione Naive Bayes: {'SPAM' if nb_crit else 'HAM'}")
    print(f"Predizione SVM:         {'SPAM' if svm_crit else 'HAM'}")

    # G. RISULTATI E ANALISI SBILANCIAMENTO
    models = {
        "Naive Bayes (Probabilistico)": nb_preds,
        "SVM (Geometrico)": svm_preds
    }

    print("\n--- Analisi Performance ---")
    for name, preds in models.items():
        b_acc = balanced_accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"Modello: {name}")
        print(f"  Balanced Accuracy: {b_acc:.4f}")
        print(f"  F1-Score:          {f1:.4f}")
    
    print("\n--- CONSIDERAZIONI SULLO SBILANCIAMENTO ---")
    print("Il Naive Bayes tende ad essere più sensibile allo sbilanciamento delle classi.")
    print("Poiché calcola il prodotto tra la probabilità della parola e la probabilità a priori della classe,")
    print("se una classe (Ham) è molto più frequente, il modello 'pende' verso di essa a meno che i dati non siano")
    print("estremamente caratterizzanti. SVM Lineare è spesso più robusto perché cerca di massimizzare il margine")
    print("tra i support vectors, focalizzandosi sugli esempi 'difficili' al confine tra le classi.")

if __name__ == "__main__":
    main()