"""
================================================================================
COSTRUZIONE DI UNA RETE RICORRENTE (RNN) PER NLP
================================================================================
Funzionamento di una una RNN. Il flusso logico è il seguente:
1. PREPARAZIONE DATI: Generiamo "frasi" (sequenze di numeri) e "sentimenti" (0 o 1).
2. ARCHITETTURA: Creiamo una catena di montaggio (il Modello) dove:
   - Input: Riceve i dati grezzi.
   - Embedding: Traduce i numeri in concetti (vettori).
   - SimpleRNN: Elabora la sequenza cercando di "ricordare" le parole precedenti.
   - Dense: Prende la decisione finale (0 o 1).
3. TRAINING: Il modello prova, sbaglia, vede l'errore e si corregge tramite l'Ottimizzatore.
4. ANALISI: Guardiamo i grafici per capire se il modello ha imparato davvero.
================================================================================
"""

import os

# --- 1. CONFIGURAZIONE DEL BACKEND ---
# Keras 3 può usare diversi "motori" (PyTorch, TensorFlow, JAX).
# Qui diciamo a Keras di usare PyTorch per tutti i calcoli matematici pesanti.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_data(num_samples=1000, max_len=20):
    """
    Simula la creazione di un dataset per addestrare l'intelligenza artificiale.
    
    Interazione: Questi dati saranno "mangiati" dal modello durante la fase .fit().
    
    Args:
        num_samples: Quante frasi simulate vogliamo creare.
        max_len: Quante parole (numeri) ci sono in ogni frase.
    """
    # Generiamo numeri casuali tra 1 e 100 per simulare indici di parole in un vocabolario
    # Output: Una matrice [1000 righe, 20 colonne]
    x = np.random.randint(1, 100, size=(num_samples, max_len))
    
    # Creiamo una regola logica che il modello dovrà scoprire:
    # Se la somma degli indici è pari, l'etichetta è 1 (positivo), altrimenti 0 (negativo).
    # .astype(int) trasforma i valori Booleani (True/False) in 1 e 0.
    y = (np.sum(x, axis=1) % 2 == 0).astype(int)
    
    # Separiamo i dati: l'80% per studiare (Train), il 20% per interrogarlo dopo (Validation).
    split = int(num_samples * 0.8)
    return (x[:split], y[:split]), (x[split:], y[split:])

def build_rnn_model(max_len, vocab_size, rnn_units=64):
    """
    Costruisce la struttura del cervello artificiale (Il Modello).
    
    Interazione tra Layer:
    - Input riceve i dati -> Embedding trasforma i dati -> RNN elabora la sequenza -> Dense decide.
    """
    
    # --- STEP A: DEFINIZIONE DELL'INGRESSO ---
    # Definiamo la "forma" del dato in entrata: liste lunghe 'max_len' (20 nel nostro caso).
    inputs = layers.Input(shape=(max_len,))
    
    # --- STEP B: IL TRADUTTORE (EMBEDDING) ---
    # Poiché l'IA non capisce gli interi, questo layer associa ogni numero a un vettore di 32 cifre decimali.
    # Queste cifre cambieranno durante il training per raggruppare "parole" simili.
    embedding = layers.Embedding(input_dim=vocab_size, output_dim=32)(inputs)
    
    # --- STEP C: IL CUORE DELLA RNN (SIMPLERNN) ---
    # Qui avviene la magia della ricorsione. Il layer legge la sequenza una "parola" alla volta.
    # - 'units=64': Il numero di neuroni (memoria interna).
    # - 'return_sequences=False': Ci restituisce solo il risultato finale della lettura (Many-to-One).
    # - 'kernel_initializer/recurrent_initializer': Impostano come devono nascere i pesi (i "neuroni") 
    #   per evitare che il modello parta con valori troppo alti o troppo bassi.
    rnn_output = layers.SimpleRNN(units=rnn_units, 
                                  activation="tanh", 
                                  return_sequences=False,
                                  kernel_initializer="glorot_uniform",
                                  recurrent_initializer="orthogonal")(embedding)
    
    # --- STEP D: IL DECISORE FINALE (DENSE) ---
    # Un singolo neurone finale. L'attivazione 'sigmoid' schiaccia il risultato tra 0 e 1.
    # Se il valore è vicino a 1, il modello dirà "Positivo", se vicino a 0 "Negativo".
    outputs = layers.Dense(1, activation="sigmoid")(rnn_output)
    
    # Creiamo l'oggetto Modello che unisce l'inizio (inputs) alla fine (outputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Esempio_RNN_Studenti")
    
    # --- STEP E: REGOLE DI ADDESTRAMENTO (COMPILAZIONE) ---
    # - 'learning_rate=0.001': La velocità con cui il modello corregge i propri errori.
    # - 'clipnorm=1.0': -> Se il modello riceve un segnale troppo forte che potrebbe 
    #   confonderlo (Exploding Gradient), lo "taglia" a 1.0 per rimanere stabile.
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy", # Funzione di errore per problemi a due scelte (0/1)
        metrics=["accuracy"]        # Vogliamo monitorare la percentuale di risposte esatte
    )
    
    return model

def plot_history(history):
    """
    Trasforma il "diario dell'addestramento" in grafici leggibili.
    
    Interazione: Prende l'input dall'oggetto 'history' creato dal comando .fit().
    """
    # Estraiamo i dati di accuratezza e perdita registrati epoche dopo epoche
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    # Creiamo una finestra con due grafici affiancati
    plt.figure(figsize=(12, 5))

    # GRAFICO 1: ERRORE (LOSS) - Più scende, meglio è!
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, 'b-', label='Errore Training')
    plt.plot(epochs_range, val_loss, 'r-', label='Errore Validazione')
    plt.title('Quanto sbaglia il modello?')
    plt.xlabel('Epoche (Quante volte ha riletto i dati)')
    plt.ylabel('Loss')
    plt.legend()

    # GRAFICO 2: PRECISIONE (ACCURACY) - Più sale, meglio è!
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, 'b-', label='Precisione Training')
    plt.plot(epochs_range, val_acc, 'r-', label='Precisione Validazione')
    plt.title('Quanto indovina il modello?')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# ================================================================================
# FASE DI ESECUZIONE (IL MAIN)
# ================================================================================

if __name__ == "__main__":
    # 1. Settiamo le costanti
    MAX_LEN = 20        # Lunghezza delle frasi
    VOCAB_SIZE = 100    # Numero di parole diverse disponibili
    
    # 2. Generiamo i dati (Interazione: Dati -> Modello)
    print(">>> 1. Sto creando il set di dati simulato...")
    (x_train, y_train), (x_val, y_val) = generate_dummy_data(max_len=MAX_LEN)
    
    # 3. Costruiamo il modello (Interazione: Architettura -> Configurazione)
    print(">>> 2. Sto assemblando la rete neurale (SimpleRNN)...")
    model = build_rnn_model(max_len=MAX_LEN, vocab_size=VOCAB_SIZE)
    model.summary() # Questa tabella mostra quanti "neuroni" (parametri) verranno addestrati
    
    # 4. Iniziamo lo studio (Training)
    print(">>> 3. Inizia l'addestramento! Il modello sta cercando i pattern nei dati...")
    
    # Il Callback EarlyStopping interrompe lo studio se il modello inizia a imparare a memoria (Overfitting)
    # invece di capire i concetti generali. 'patience=3' significa: aspetta 3 volte se non migliori.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # .fit() è il cuore: Qui il modello legge x_train, prova a indovinare y_train e si corregge.
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val), # Gli diamo dati che non ha mai visto per metterlo alla prova
        epochs=30,          # Numero massimo di volte che può leggere tutto il dataset
        batch_size=32,      # Quanti esempi legge prima di correggersi (un "pacchetto")
        callbacks=[early_stop],
        verbose=1           # Mostra il progresso riga per riga nella console
    )
    
    # 5. Visualizziamo i risultati (Interazione: Storia -> Grafici)
    print(">>> 4. Fine addestramento. Sto generando i grafici finali...")
    plot_history(history)