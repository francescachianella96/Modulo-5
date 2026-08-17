import os

# =================================================================
# 1. SETUP DELL'AMBIENTE (MOTORE DI CALCOLO)
# =================================================================
# Indichiamo a Keras di usare PyTorch come "muscoli" per i calcoli.
# Deve essere la prima operazione assoluta dello script.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

"""

 =================================================================
 BIDIRECTIONAL LSTM (Bi-LSTM) SU DATASET REUTERS
 =================================================================

MAPPA LOGICA DEL CODICE:
1. PREPARAZIONE: Trasformiamo le notizie (testo) in sequenze di numeri.
2. ARCHITETTURA: Costruiamo una rete che legge "avanti e indietro".
3. ADDESTRAMENTO: Il modello impara a collegare parole e argomenti.
4. TEST REALE: Proviamo il modello su news mai viste, decodificando i numeri in parole.

VANTAGGIO BI-LSTM: 
In una news come "Il tasso di interesse della FED...", la parola 'FED' alla fine 
chiarisce che 'tasso' si riferisce alla finanza e non a un animale. La Bi-LSTM
guarda l'intera frase prima di decidere.
"""

# --- CONFIGURAZIONI GLOBALI ---
VOCABOLARIO_SIZE = 10000  # Usiamo solo le 10.000 parole più comuni al mondo
LUNGHEZZA_MAX = 150       # Ogni news viene standardizzata a 150 parole (Taglio o Padding)

# =================================================================
# 2. GESTIONE DEI DATI (IL "CARBURANTE")
# =================================================================
"""
NOTIZIE SUL DATASET REUTERS:
Il dataset è composto da 11.228 lanci d'agenzia (newswires) provenienti 
dalla Reuters, una delle principali agenzie di stampa mondiali.

PUNTI CHIAVE:
- CATEGORIE: 46 argomenti diversi (es. finanze, agricoltura, energia).
- FORMATO: Ogni notizia è già stata pre-elaborata: le parole sono state sostituite 
  da numeri interi che rappresentano la loro frequenza nel dataset.
  - Il numero 10 rappresenta la decima parola più frequente.
  - Questo risparmia tempo nella pulizia dei testi reali.
- SUDDIVISIONE: Circa 8.982 esempi per l'addestramento e 2.246 per il test.
"""

def prepara_dataset():
    """
    Estrae le notizie dal dataset Reuters e le rende uniformi per la rete neurale.
    """
    print("[1] Caricamento Dataset Reuters (46 categorie tematiche)...")
    
    # Caricamento: Keras ci fornisce già gli ID numerici delle parole.
    # Esempio: "apple" diventa 42, "market" diventa 105.
    (x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=VOCABOLARIO_SIZE)
    
    # INTERAZIONE: pad_sequences trasforma liste di lunghezze diverse in una matrice rettangolare.
    # Se una news ha 50 parole, aggiungiamo 100 zeri (Padding).
    x_train = keras.utils.pad_sequences(x_train, maxlen=LUNGHEZZA_MAX)
    x_test = keras.utils.pad_sequences(x_test, maxlen=LUNGHEZZA_MAX)
    
    # INTERAZIONE: to_categorical trasforma l'indice della categoria (es. 3) in un vettore unitario.
    # Categoria 3 -> [0, 0, 0, 1, 0, ... 0] (46 posizioni totali).
    y_train = keras.utils.to_categorical(y_train, 46)
    y_test = keras.utils.to_categorical(y_test, 46)
    
    return (x_train, y_train), (x_test, y_test)

# =================================================================
# 3. COSTRUZIONE DEL MODELLO (IL "CERVELLO")
# =================================================================
def build_bilstm_classifier():
    """
    Definisce come le informazioni fluiscono dall'input alla decisione finale.
    """
    # Definiamo la forma del dato in ingresso (150 numeri interi)
    inputs = keras.Input(shape=(LUNGHEZZA_MAX,), name="Ingresso_News")

    # EMBEDDING (IL TRADUTTORE)
    # Trasforma ogni numero ID in un vettore di 128 caratteristiche (significato).
    # Interazione: Trasforma INDICI (interi) -> SPAZIO SEMANTICO (float).
    x = layers.Embedding(VOCABOLARIO_SIZE, 128, name="Spazio_Semantico")(inputs)

    # BIDIRECTIONAL LSTM (IL DOPPIO SGUARDO)
    # Layers.Bidirectional avvolge una LSTM standard e ne crea due:
    # 1. Forward LSTM: Legge la frase da sinistra a destra.
    # 2. Backward LSTM: Legge la frase da destra a sinistra.
    # Interazione: Fonde i due contesti in un unico vettore di memoria (64+64 = 128 unità).
    x = layers.Bidirectional(layers.LSTM(64), name="Memoria_Bidirezionale")(x)
    
    # DROPOUT (IL FILTRO ANTI-MEMORIA)
    # Spegne casualmente il 30% dei neuroni per forzare la rete a non imparare i dati a memoria.
    x = layers.Dropout(0.3)(x)
    
    # DENSE (IL RAGIONAMENTO FINALE)
    # Strato con 64 neuroni per elaborare i significati estratti dalla Bi-LSTM.
    x = layers.Dense(64, activation="relu")(x)
    
    # OUTPUT (LA SCELTA FINALE)
    # 46 neuroni (uno per ogni categoria). 'softmax' garantisce che la somma delle probabilità sia 1.
    outputs = layers.Dense(46, activation="softmax", name="Probabilita_Categorie")(x)

    # ASSEMBLAGGIO: Colleghiamo inizio e fine
    model = keras.Model(inputs, outputs, name="Classificatore_BiLSTM_Reuters")
    
    model.compile(
        optimizer="adamw",           # Algoritmo che aggiorna i pesi (il più moderno nel 2026)
        loss="categorical_crossentropy", # Funzione che punisce gli errori di categoria
        metrics=["accuracy"]         # Vogliamo vedere la percentuale di risposte corrette
    )
    return model

# =================================================================
# 4. ESECUZIONE E TEST REAL-TIME
# =================================================================
def main():
    print("-" * 60)
    print("DEMO: BI-LSTM ALL'OPERA SULLE NEWS REUTERS")
    print("-" * 60)
    
    # STEP 1: Preparazione dati (Interazione con la memoria RAM)
    (x_train, y_train), (x_test, y_test) = prepara_dataset()
    
    # STEP 2: Creazione architettura (Interazione con Keras/PyTorch)
    model = build_bilstm_classifier()
    model.summary() # Mostra la struttura e il numero di parametri (pesi) da imparare
    
    # STEP 3: Addestramento (Il "Training" effettivo)
    print("\n[HINT] La rete sta leggendo 9.000 notizie per imparare i temi...")
    model.fit(
        x_train, y_train,
        epochs=10,            # Quante volte la rete rilegge tutto il dataset
        batch_size=128,       # Quante news guarda insieme prima di aggiornare i pesi
        validation_split=0.1, # Usa il 10% per auto-valutarsi durante l'apprendimento
        verbose=1             # Mostra la barra di progresso
    )
    
    # STEP 4: COLLAUDO SUL CAMPO (Inference)
    print("\n" + "="*50)
    print("VERIFICA: IL MODELLO ANALIZZA NOTIZIE REALI")
    print("="*50)

    # Strumenti per decodificare: serve per tornare dai numeri alle parole umane
    word_index = keras.datasets.reuters.get_word_index()
    reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])

    # Analizziamo i primi 3 esempi del set di test (mai visti prima dal modello)
    for i in range(3):
        sample = x_test[i:i+1] # Estraiamo una singola riga di dati
        prediction = model.predict(sample, verbose=0)
        
        pred_idx = np.argmax(prediction) # L'indice della probabilità più alta
        real_idx = np.argmax(y_test[i])   # L'indice reale salvato nel dataset
        
        # Ricostruiamo la frase (saltando i primi 3 indici di sistema di Keras)
        testo = ' '.join([reverse_word_index.get(index - 3, '?') for index in x_test[i]])
        testo_pulito = testo.replace('?', '').strip()[-130:] # Prendiamo la parte finale significativa
        
        print(f"\nNEWS #{i+1}: ...{testo_pulito}")
        print(f"-> PREDIZIONE: Categoria {pred_idx} (Confidenza: {np.max(prediction):.2%})")
        print(f"-> REALTA':    Categoria {real_idx}")
        
    print("\n[FINISH] Esperimento completato. La Bi-LSTM ha 'capito' il contesto globale.")

if __name__ == "__main__":
    main()

# =================================================================
# GUIDA TECNICA (FLUSSO LOGICO) PER LO STUDENTE
# =================================================================
# 1. FLUSSO DATI: Notizia (Testo) -> Liste ID -> Matrice Padding -> Embedding (Vettori) 
#    -> Bi-LSTM (Memoria) -> Dense (Decisione) -> Softmax (Probabilità).
# 2. PERCHÉ LA BI-LSTM? Perché nel giornalismo (Reuters), il soggetto di una frase 
#    lunga viene spesso chiarito solo alla fine. Leggere in entrambi i sensi
#    evita che la rete "si perda" durante il tragitto.
# 3. VERSO LA TRADUZIONE: In un modello di traduzione, la Bi-LSTM qui usata 
#    è l'ENCODER: il suo compito è creare una "mappa mentale" perfetta della 
#    frase sorgente prima di passare la palla al generatore di testo (Decoder).