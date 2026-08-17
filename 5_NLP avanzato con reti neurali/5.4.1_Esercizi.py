"""
============================================================
CONFRONTO LSTM VS LOGISTIC REGRESSION (SENTIMENT ANALYSIS)
============================================================
In questo esercizio confrontiamo la capacità di una rete neurale 
ricorrente (LSTM) di gestire la negazione ("non è ...") rispetto 
a un modello statistico classico (Logistic Regression).
"""

import os

# Impostiamo il backend di Keras a PyTorch prima di caricare Keras
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. PREPARAZIONE DATI
print("--- Caricamento Dati IMDb ---")
max_features = 10000  # Numero di parole più frequenti da considerare
maxlen = 100         # Lunghezza massima di ogni recensione

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Padding: portiamo tutte le sequenze alla stessa lunghezza
x_train_pad = pad_sequences(x_train, maxlen=maxlen)
x_test_pad = pad_sequences(x_test, maxlen=maxlen)

# Per la Logistic Regression, abbiamo bisogno del testo originale (o una sua ricostruzione)
# per applicare TF-IDF, poiché lavora meglio su frequenze di parole.
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text_indices):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in text_indices])

print("Decodifica testi per Baseline...")
x_train_text = [decode_review(x) for x in x_train]
x_test_text = [decode_review(x) for x in x_test]

# 2. MODELLO BASELINE: LOGISTIC REGRESSION (TF-IDF)
print("\n--- Training Logistic Regression (Baseline) ---")
tfidf = TfidfVectorizer(max_features=max_features)
x_train_tfidf = tfidf.fit_transform(x_train_text)
x_test_tfidf = tfidf.transform(x_test_text)

lr_model = LogisticRegression()
lr_model.fit(x_train_tfidf, y_train)
lr_acc = lr_model.score(x_test_tfidf, y_test)
print(f"Accuratezza Logistic Regression: {lr_acc:.4f}")

# 3. MODELLO LSTM (64 unità)
def build_lstm(units=64):
    model = keras.Sequential([
        layers.Embedding(input_dim=max_features, output_dim=100, name="Embedding"),
        layers.LSTM(units, name=f"LSTM_{units}"),
        layers.Dense(1, activation="sigmoid", name="Output")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

print("\n--- Training LSTM (64 unità) ---")
model_lstm_64 = build_lstm(64)
history_64 = model_lstm_64.fit(
    x_train_pad, y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.2,
    verbose=1
)

# 4. SFIDA: LSTM (128 unità)
print("\n--- Training LSTM (128 unità) - Sfida ---")
model_lstm_128 = build_lstm(128)
history_128 = model_lstm_128.fit(
    x_train_pad, y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.2,
    verbose=1
)

# 5. TEST SPECIFICO: "NON È AGGETTIVO"
# Creiamo test case per verificare la gestione della negazione.
# Nota: IMDb è in inglese, usiamo "is not good" vs "is not bad"
custom_texts = [
    "this movie is not good",  # Negativo
    "this movie is not bad",   # Positivo (struttura complessa per modelli statistici)
    "the acting was not great", # Negativo
    "i did not hate it"        # Positivo (negazione di un termine negativo)
]

def prepare_custom_data(texts):
    # Trasforma in indici compatibili con IMDb
    sequences = []
    for text in texts:
        seq = [word_index.get(w, 2) + 3 for w in text.split()]
        sequences.append([s if s < max_features else 2 for s in seq])
    return pad_sequences(sequences, maxlen=maxlen)

custom_seq = prepare_custom_data(custom_texts)
custom_tfidf = tfidf.transform(custom_texts)

print("\n--- Risultati Test Negazioni ---")
preds_lr = lr_model.predict(custom_tfidf)
preds_lstm_64 = (model_lstm_64.predict(custom_seq) > 0.5).astype(int).flatten()

for i, text in enumerate(custom_texts):
    print(f"Frase: '{text}'")
    print(f"  - Logistic Regression: {'Positivo' if preds_lr[i] == 1 else 'Negativo'}")
    print(f"  - LSTM (64):           {'Positivo' if preds_lstm_64[i] == 1 else 'Negativo'}")

# CONCLUSIONI
print("\n--- Analisi Finale ---")
val_acc_64 = history_64.history['val_accuracy'][-1]
val_acc_128 = history_128.history['val_accuracy'][-1]

print(f"Accuratezza Validazione LSTM 64:  {val_acc_64:.4f}")
print(f"Accuratezza Validazione LSTM 128: {val_acc_128:.4f}")

if val_acc_128 > val_acc_64 + 0.01:
    print("Il raddoppio delle unità ha migliorato sensibilmente la performance.")
elif abs(val_acc_128 - val_acc_64) < 0.01:
    print("Il raddoppio delle unità non ha portato benefici significativi (possibile saturazione).")
else:
    print("Il modello con 128 unità mostra segni di overfitting (val_accuracy inferiore o instabile).")