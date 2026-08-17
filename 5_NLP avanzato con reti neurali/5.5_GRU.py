"""
================================================================================
CLASSIFICAZIONE MULTI-CLASSE CON GATED RECURRENT UNITS (GRU)
================================================================================

OBIETTIVO DEL CODICE:
Dimostrare come una GRU riceve testo "grezzo", lo elabora in modo sequenziale
e decide a quale categoria appartiene (Tecnologia, Sport o Business).

INTERAZIONE FRA I COMPONENTI:
1. Dati (Testi) -> 2. TextVectorization (Numeri) -> 3. Embedding (Vettori) 
-> 4. GRU (Memoria Sequenziale) -> 5. Dense (Classificazione finale)
================================================================================
"""

import os

# [RIGA CRITICA] Impostiamo il backend PRIMA di importare Keras.
# Questo dice a Keras di usare PyTorch per tutti i calcoli matematici pesanti.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

def get_dataset():
    """
    Crea un piccolo dataset di esempio per simulare un problema reale.
    
    Interazione: Questa funzione fornisce la materia prima (testo e etichette) 
    che verrà poi 'masticata' dal preprocessore e dal modello.
    """
    # Lista di tuple: (Frase, Categoria_Numerica)
    # 0 = Tecnologia, 1 = Sport, 2 = Business
    data = [
        ("Il nuovo processore quantistico batte ogni record di velocità", 0),
        ("L'intelligenza artificiale generativa rivoluziona il software", 0),
        ("La finale di campionato si giocherà allo stadio olimpico", 1),
        ("Il tennista ha vinto il torneo del Grande Slam", 1),
        ("Le azioni in borsa sono salite dopo il report trimestrale", 2),
        ("Startup innovativa riceve un finanziamento record da capitalisti", 2),
        ("Aggiornamento sistema operativo disponibile per il download", 0),
        ("La squadra di calcio ha cambiato allenatore", 1),
        ("Inflazione in calo secondo i dati della banca centrale", 2)
    ]
    
    # Separiamo le frasi dalle etichette
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    # [SLIDE 12] Trasformiamo i numeri (es. 0, 1) in vettori "One-hot" (es. [1, 0, 0]).
    # Questo è necessario perché l'ultimo layer del modello sputa fuori 3 probabilità.
    labels_one_hot = keras.utils.to_categorical(labels, num_classes=3)
    
    return texts, labels_one_hot, ["Tecnologia", "Sport", "Business"]

def build_gru_model(vocab_size, num_classes, max_len):
    """
    Crea l'architettura della rete neurale (il "cervello").
    
    Argomenti:
        vocab_size: quante parole diverse il modello può conoscere.
        num_classes: in quante categorie vogliamo dividere il testo.
        max_len: quante parole al massimo leggiamo per ogni frase.
    """
    # 1. INPUT LAYER: Definisce la forma del dato in ingresso. 
    # Aspettiamo una sequenza di 'max_len' numeri interi (indici delle parole).
    inputs = layers.Input(shape=(max_len,), name="Ingresso_Numerico")
    
    # 2. EMBEDDING LAYER: Il "traduttore semantico".
    # Trasforma ogni numero di parola in un vettore di 64 numeri "significativi".
    # Le parole con significato simile finiranno vicine in questo spazio matematico.
    embedding = layers.Embedding(
        input_dim=vocab_size, 
        output_dim=64, 
        name="Dizionario_Vettoriale"
    )(inputs)
    
    # 3. GRU LAYER [CUORE DELLA LEZIONE]:
    # Legge i vettori uno dopo l'altro. Grazie ai suoi 2 Gate (Update e Reset),
    # decide cosa ricordare delle parole precedenti per capire il senso della frase.
    # units=32: è la "capienza" della sua memoria di lavoro.
    # dropout: serve a non far "imparare a memoria" il dataset (generalizzazione).
    gru_out = layers.GRU(
        units=32, 
        dropout=0.2, 
        name="Motore_GRU"
    )(embedding)
    
    # 4. BATCH NORMALIZATION: Stabilizzatore.
    # "Pulisce" e normalizza i dati in uscita dalla GRU per aiutare i layer successivi.
    x = layers.BatchNormalization()(gru_out)
    
    # 5. DENSE LAYER: Analisi finale.
    # Prende la memoria della GRU e cerca pattern specifici per le categorie.
    x = layers.Dense(16, activation="relu", name="Analisi_Densa")(x)
    
    # 6. OUTPUT LAYER (SOFTMAX): La decisione finale.
    # Sputa fuori 3 numeri (es: 0.05, 0.90, 0.05) che sommano a 1.
    # Rappresentano la probabilità per ogni classe (Tecnologia, Sport, Business).
    outputs = layers.Dense(num_classes, activation="softmax", name="Probabilita_Classi")(x)
    
    # Creiamo l'oggetto modello collegando Ingressi e Uscite
    model = keras.Model(inputs=inputs, outputs=outputs, name="Cervello_GRU")
    
    # COMPILAZIONE: Diciamo al modello come imparare dagli errori.
    # Loss 'categorical_crossentropy': la punizione per quando sbaglia a indovinare la classe.
    # Optimizer 'adam': l'algoritmo che corregge i pesi internamente.
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    """
    Sequenza operativa principale: collega dati, modello e addestramento.
    """
    # PASSO A: Carichiamo i dati grezzi
    texts, labels, class_names = get_dataset()
    
    # PASSO B: TRASFORMAZIONE TESTO -> NUMERI (Preprocessing)
    # Creiamo un layer che pulisce il testo e assegna un numero a ogni parola.
    max_features = 1000  # Vocabolario massimo
    max_len = 10         # Lunghezza singola frase
    
    vectorize_layer = layers.TextVectorization(
        max_tokens=max_features,
        output_mode="int", # Vogliamo numeri interi
        output_sequence_length=max_len # Tutte le frasi avranno la stessa lunghezza
    )
    
    # Il layer deve "leggere" i nostri testi per creare il vocabolario (mapping parola->numero)
    vectorize_layer.adapt(np.array(texts))
    
    # Trasformiamo effettivamente le nostre frasi in matrici di numeri
    x_train = vectorize_layer(np.array(texts))
    
    # PASSO C: COSTRUZIONE
    # Generiamo il modello usando la funzione definita sopra
    model = build_gru_model(max_features, len(class_names), max_len)
    
    # Mostriamo la "Tabella dei Pesi" (Summary). 
    # Notate come i parametri della GRU siano circa il 25% in meno di una LSTM!
    model.summary()
    
    # PASSO D: ADDESTRAMENTO
    # Il modello guarda i dati (x_train) e le risposte (labels) per 50 volte (epochs)
    print("\n--- INIZIO ADDESTRAMENTO ---")
    model.fit(x_train, labels, epochs=50, verbose=0) # verbose=0 per pulizia output
    print("Addestramento completato.")
    
    # PASSO E: INFERENZA (Prova su una frase mai vista)
    nuova_frase = ["Il calciatore ha segnato un gol incredibile"]
    
    # 1. Trasformiamo la frase in numeri (lo stesso processo usato per il training)
    vettore_test = vectorize_layer(np.array(nuova_frase))
    
    # 2. Chiediamo al modello una previsione
    previsione = model.predict(vettore_test, verbose=0)
    
    # 3. Prendiamo l'indice con la probabilità più alta (es. se [0.1, 0.8, 0.1] prendiamo 1)
    indice_predetto = np.argmax(previsione)
    probabilita = np.max(previsione)
    
    print(f"\nFRASE DI TEST: '{nuova_frase[0]}'")
    print(f"RISULTATO: {class_names[indice_predetto]} (Confidenza: {probabilita*100:.2f}%)")

if __name__ == "__main__":
    main()

# ==============================================================================
# RIASSUNTO DELLE OPERAZIONI RIGA PER RIGA PER CHI LEGGE PER LA PRIMA VOLTA:
# ==============================================================================
# 1. Import: Carichiamo gli strumenti (Keras per la rete, Numpy per i numeri).
# 2. get_dataset(): Prepariamo gli esempi. Fondamentale il to_categorical per
#    adeguare le etichette al formato di uscita della rete (softmax).
# 3. build_gru_model(): Progettiamo la "macchina". 
#    - Input riceve i numeri.
#    - Embedding dà loro un significato vettoriale.
#    - GRU capisce l'ordine delle parole.
#    - Dense/Softmax decidono la categoria.
# 4. TextVectorization.adapt(): Questa riga crea il dizionario interno. 
#    Senza questa, il modello non saprebbe che "calcio" corrisponde, ad esempio, al numero 42.
# 5. model.fit(): È il momento in cui i neuroni si attivano e si aggiustano
#    per minimizzare l'errore tra ciò che dicono e la realtà.
# 6. predict(): Usiamo il cervello ormai addestrato su dati nuovi.
# ==============================================================================