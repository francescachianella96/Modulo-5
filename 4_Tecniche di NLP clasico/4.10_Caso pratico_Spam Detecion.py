"""
================================================================================
ESEMPIO PRATICO: PIPELINE DI SPAM DETECTION 
================================================================================
Questo script rappresenta una "Pipeline" completa. In ambito professionale, una 
pipeline è un insieme di passaggi sequenziali che trasformano i dati grezzi 
(SMS sporchi) in una decisione intelligente (Spam o Ham).

INTERAZIONI TRA LE COMPONENTI:
1. PREPROCESSING: Pulisce il testo (clean_text) usando dizionari di slang.
2. VETTORIZZAZIONE: Trasforma parole in numeri (TF-IDF) affinché i modelli possano "leggerle".
3. TORNEO DEI MODELLI: Tre diversi algoritmi (NB, SVM, Keras) competono per 
   lo stesso obiettivo, permettendoci di scegliere il migliore.
4. METRICHE: Misuriamo il successo non solo con l'accuratezza classica, ma con il
   Balanced Accuracy (cruciale quando gli Spam sono pochi rispetto agli SMS normali).
"""

import os

# CONFIGURAZIONE BACKEND: Nel 2026, Keras funge da "ponte" sopra diversi motori (PyTorch, JAX, TF).
# Impostiamo PyTorch per sfruttare la sua velocità e l'ecosistema di ricerca.
os.environ["KERAS_BACKEND"] = "torch"

import re
import numpy as np
import pandas as pd
import keras
from keras import layers, ops
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from nltk.stem import PorterStemmer

# --- 1. COMPONENTE: GESTIONE DELLA CONOSCENZA (DICTIONARY MAPPING) ---

def get_slang_map():
    """
    Ritorna una mappa di traduzione per il linguaggio SMS.
    Gli SMS usano abbreviazioni che i modelli non capirebbero. Traducendo 'u' in 'you',
    permettiamo al modello di trovare schemi comuni tra diversi messaggi.
    """
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

# --- 2. COMPONENTE: IL CHIRURGO DEL TESTO (PREPROCESSING) ---

def clean_text(text, stemmer=PorterStemmer()):
    """
    Esegue la "pulizia profonda" del testo.
    Ogni riga qui sotto rimuove il 'rumore' e mantiene solo il 'segnale'.
    """
    # 1. Normalizzazione: 'SPAM' e 'spam' devono essere considerati la stessa parola.
    text = text.lower()
    
    # 2. Espansione Slang: Applichiamo il dizionario definito sopra.
    slang_map = get_slang_map()
    words = text.split()
    text = " ".join([slang_map.get(w, w) for w in words])
    
    # 3. Tokenizzazione Speciale: Sostituiamo URL e numeri con 'placeholder'.
    # Questo dice al modello: "Qui c'era un link", senza confonderlo con l'indirizzo specifico.
    text = re.sub(r'http[s]?://\S+', 'url_token', text)
    text = re.sub(r'\d+', 'num_token', text)
    
    # 4. Filtro Caratteri: Rimuoviamo simboli inutili ma teniamo '!' (spesso marchio di spam).
    text = re.sub(r'[^a-zA-Z!\s]', '', text)
    
    # 5. Stemming: Riduce le parole alla radice (es. 'calling' -> 'call', 'calls' -> 'call').
    # Riduce drasticamente il numero di parole uniche che il modello deve imparare.
    text = " ".join([stemmer.stem(w) for w in text.split()])
    
    return text

# --- 3. COMPONENTE: L'ARCHITETTO NEURALE (DEEP LEARNING) ---

def build_keras_classifier(input_shape):
    """
    Crea un classificatore 'Deep' (Rete Neurale).
    A differenza di un modello statistico semplice, questo può imparare combinazioni
    complesse tra le parole grazie ai suoi 'strati nascosti'.
    """
    # INPUT: Definisce quanto è grande il vettore che entra (numero di feature TF-IDF).
    inputs = keras.Input(shape=(input_shape,), name="tfidf_input")
    
    # STRATO DENSE (Hidden): 16 neuroni catturano le interazioni tra le parole.
    # 'relu' è l'attivazione standard che permette alla rete di modellare problemi non lineari.
    x = layers.Dense(16, activation="relu")(inputs)
    
    # DROPOUT: Spegne casualmente il 20% dei neuroni durante il training.
    # Impedisce che il modello "impari a memoria" i dati (Overfitting).
    x = layers.Dropout(0.2)(x)
    
    # OUTPUT: Un solo neurone con attivazione 'sigmoid'.
    # Trasforma l'uscita in una probabilità tra 0 (Ham) e 1 (Spam).
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="SpamDetector_Keras")
    
    # COMPILAZIONE: Scegliamo come il modello deve imparare (Adam) e come misurare l'errore (Binary Crossentropy).
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

# --- 4. IL MOTORE DI ESECUZIONE (LOGICA DI BUSINESS) ---

def main():
    # --- A. CARICAMENTO DATASET REALE ---
    # Utilizziamo il dataset "SMS Spam Collection" (UCI Machine Learning Repository)
    # Caricato da un link raw affidabile per garantire l'esecuzione immediata.
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    
    print(f"--- Download Dataset in corso da: {url} ---")
    try:
        # Il dataset è un TSV (Tab Separated Values) senza header, con colonne: label e text
        df = pd.read_csv(url, sep='\t', names=['label_raw', 'text'])
        print(f"Dataset caricato con successo! Numero di messaggi: {len(df)}")
    except Exception as e:
        print(f"Errore nel download: {e}. Uso un piccolo set di emergenza.")
        df = pd.DataFrame({
            'text': ["Free money now!", "Hello how are you?", "Win a prize", "Meeting at 5"],
            'label_raw': ['spam', 'ham', 'spam', 'ham']
        })

    # Mappatura delle etichette: 'ham' -> 0, 'spam' -> 1
    df['label'] = df['label_raw'].map({'ham': 0, 'spam': 1})

    # B. PRIMO STEP: PULIZIA (Applicazione della funzione clean_text riga per riga)
    print("--- Inizio Preprocessing (Richiede qualche secondo per 5000+ messaggi)... ---")
    df['clean_text'] = df['text'].apply(clean_text)

    # C. SECONDO STEP: TRASFORMAZIONE NUMERICA (TF-IDF)
    # Nota: su un dataset reale, 1000 feature iniziano a mostrare le differenze tra i modelli.
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label'].values

    # D. TERZO STEP: SEPARAZIONE (TRAIN & TEST)
    # Alleniamo i modelli sull'80% dei dati e poi li mettiamo alla prova sul restante 20%.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # E. IL TORNEO (Confronto tra filosofie diverse)
    
    # 1. NAIVE BAYES: Semplice e veloce, basato sulla probabilità statistica.
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)

    # 2. SVM (Support Vector Machine): Cerca di "disegnare" il confine migliore tra le classi.
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)

    # 3. KERAS (Deep Learning): La nostra rete neurale definita prima.
    keras_model = build_keras_classifier(X_train.shape[1])
    # Alleniamo per 10 epoche: il modello guarda i dati 10 volte per perfezionarsi.
    keras_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    keras_probs = keras_model.predict(X_test)
    # Se la probabilità è > 0.5 diciamo che è Spam (1), altrimenti Ham (0).
    keras_preds = (keras_probs > 0.5).astype(int).flatten()

    # F. VALUTAZIONE FINALE
    models = {
        "Naive Bayes (Statistico)": nb_preds,
        "SVM (Geometrico)": svm_preds,
        "Keras (Neurale)": keras_preds
    }

    print("\n--- Risultati del Torneo dei Modelli ---")
    for name, preds in models.items():
        # Balanced Accuracy: Corregge il punteggio se abbiamo molti più messaggi Ham che Spam.
        b_acc = balanced_accuracy_score(y_test, preds)
        # F1-Score: Bilancia precisione (non sbagliare Spam) e richiamo (trovarli tutti).
        f1 = f1_score(y_test, preds)
        print(f"Modello: {name}")
        print(f"  Balanced Accuracy: {b_acc:.4f}")
        print(f"  F1-Score:          {f1:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()