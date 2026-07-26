"""
PIPELINE DI VECTORIZATION CON SCIKIT-LEARN
------------------------------------------
Questo script dimostra il workflow per trasformare testo non strutturato
in rappresentazioni numeriche (vettori) utilizzabili dai modelli di Machine Learning.

Concetti chiave:
1. Tokenizzazione: Divisione del testo in singole parole o gruppi di parole (n-grammi).
2. Vocabolario: Mappatura univoca tra parole e indici numerici.
3. Vettorizzazione: Trasformazione di documenti in righe di una matrice sparsa.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib  # Standard per la serializzazione efficiente di oggetti scikit-learn

# ==========================================
# 1. DEFINIZIONE DEL CORPUS (Dataset di Testo)
# ==========================================
# Il "Corpus" è la collezione completa dei documenti che useremo per "istruire" 
# il nostro vettorizzatore. Da qui verrà estratto il vocabolario globale.
corpus_train = [
    "L'intelligenza artificiale sta cambiando il mondo.",
    "Il deep learning è una sottocategoria dell'intelligenza artificiale.",
    "Le reti neurali sono alla base dei modelli di deep learning.",
    "Python è il linguaggio principale per la data science."
]

# =============================================================
# 2. COUNT-VECTORIZATION (Bag of Words - BoW)
# =============================================================
# Classe: CountVectorizer
# Scopo: Conta semplicemente quante volte ogni parola appare in ogni documento.

# Parametro ngram_range=(1, 2): 
# - Include unigrammi ("intelligenza") 
# - Include bigrammi ("intelligenza artificiale")
# Utile per catturare il contesto locale dove l'ordine delle parole conta.
count_vec = CountVectorizer(ngram_range=(1, 2))

# METODO fit_transform:
# 1. .fit(): Analizza il corpus_train e costruisce il vocabolario (ID <-> Parola).
# 2. .transform(): Converte ogni frase in un vettore di conteggi basato sul vocabolario creato.
# Risultato: Una matrice SciPy CSR (Compressed Sparse Row) per risparmiare memoria (piena di zeri).
X_counts = count_vec.fit_transform(corpus_train)

print(f"--- SEZIONE 1: CountVectorizer ---")
print(f"Dimensioni matrice BoW (Documenti, Termini): {X_counts.shape}")
# get_feature_names_out(): Estrae la lista ordinata delle parole che compongono le colonne della matrice.
print(f"Top 5 termini nel vocabolario: {count_vec.get_feature_names_out()[:5]}")


# =============================================================
# 3. TF-IDF E OTTIMIZZAZIONE DELLE FEATURE
# =============================================================
# Classe: TfidfVectorizer
# Scopo: Pesa le parole non solo in base alla frequenza (TF), ma anche alla loro 
# rarità nel corpus (IDF - Inverse Document Frequency). Parole comuni come "il" 
# avranno peso basso; parole specifiche come "neurali" avranno peso alto.

# Parametro max_features=10:
# Seleziona solo i 10 termini con il punteggio TF-IDF più alto in tutto il dataset.
# Strategia cruciale per ridurre il "rumore" e la complessità computazionale.
tfidf_vec = TfidfVectorizer(max_features=10, stop_words=None)

# Eseguiamo la trasformazione statistica
X_tfidf = tfidf_vec.fit_transform(corpus_train)

# La matrice risultante ha righe normalizzate (L2 normalization):
# Ogni riga è un vettore la cui somma dei quadrati degli elementi è 1.
print(f"\n--- SEZIONE 2: TF-IDF ---")
print(f"Feature più rilevanti selezionate: {tfidf_vec.get_feature_names_out()}")


# =============================================================
# 4. PERSISTENZA E SERIALIZZAZIONE (Model Saving)
# =============================================================
# Perché salvare? In produzione non avremo il corpus originale. 
# Dobbiamo usare LO STESSO vocabolario e pesi IDF per trasformare i nuovi dati.
model_filename = "tfidf_vectorizer_v1.joblib"

# joblib.dump: Salva l'oggetto Python (lo stato interno del vettorizzatore) in un file binario.
joblib.dump(tfidf_vec, model_filename)
print(f"\n[INFO] Vettorizzatore 'fit' salvato in: {model_filename}")


# =============================================================
# 5. FASE DI INFERENZA (Utilizzo su nuovi dati)
# =============================================================
# Simuliamo l'applicazione del modello su un server o in una nuova sessione.

# 5.1 Caricamento dell'oggetto precedentemente addestrato
loaded_vec = joblib.load(model_filename)

# 5.2 Nuovo input (mai visto durante l'addestramento)
new_input = ["L'intelligenza artificiale usa le reti neurali."]

# 5.3 TRASFORMAZIONE (Cruciale: SENZA .fit())
# Usiamo SOLO .transform() perché:
# - Se usassimo .fit(), creeremmo un nuovo vocabolario ignorando quello di training.
# - Se l'input contiene parole nuove (es. "usa"), esse verranno ignorate per mantenere
#   la compatibilità con le dimensioni attese dal modello di ML a valle.
X_new = loaded_vec.transform(new_input)

print("\n--- SEZIONE 3: Inferenza ---")
print("Vettore numerico generato per il nuovo input (rappresentazione sparsa):")
print(X_new)
print("\nSpiegazione: Gli indici sopra corrispondono ai termini nel vocabolario caricato.")