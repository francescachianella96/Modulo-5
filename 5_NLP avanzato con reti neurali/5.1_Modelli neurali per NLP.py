"""
================================================================================
MODELLI NEURALI PER SEQUENZE (NLP)
================================================================================

Questo script spiega come le macchine imparano a leggere.
Vedremo come trasformare parole in numeri, come gestire frasi di lunghezze diverse
e perché le reti LSTM sono migliori delle semplici RNN per "ricordare" il contesto.

ARCHITETTURA LOGICA:
1. TESTO GREGGIO -> 2. VETTORIZZAZIONE (numeri) -> 3. EMBEDDING (concetti) -> 
4. LSTM (memoria) -> 5. DENSE (decisione finale).
"""

import os

# --- PASSO 0: CONFIGURAZIONE MOTORE ---
# Keras 3 è un'interfaccia universale. Qui diciamo a Keras di usare PyTorch 
# come "motore aritmetico" (backend) per eseguire i calcoli.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np

def crea_dataset_didattico():
    """
    FASE 1: PREPARAZIONE MATERIA PRIMA.
    Simuliamo delle recensioni o frasi che la rete dovrà analizzare.
    
    PERCHÉ QUESTE FRASI? 
    Hanno lunghezze diverse (da 4 a 10 parole). Questo serve a testare il PADDING.
    """
    testi = [
        "Il gatto rincorre il topo",                                     # Breve
        "Oggi il cielo è molto limpido e azzurro sopra le montagne",    # Lunga
        "Deep Learning è fantastico",                                    # Corta
        "La memoria delle reti neurali è complessa ma affascinante"     # Media
    ]
    # Etichette di target: vogliamo che la rete impari a classificarle (es. Sentiment Positivo = 1)
    etichette = np.array([1, 1, 1, 1], dtype="float32")
    
    return testi, etichette

def preprocessamento_dati(testi):
    """
    FASE 2: TRADUZIONE UMANO -> MACCHINA.
    I computer non capiscono le lettere, leggono solo numeri (tensori).
    """
    # 1. Inizializziamo il Vectorizer: agisce come un dizionario dinamico.
    # max_tokens=100: tiene solo le 100 parole più comuni.
    # output_sequence_length=12: taglia le frasi lunghe o allunga quelle corte 
    # aggiungendo zeri (PADDING) per rendere ogni "striscia" di dati lunga 12.
    vectorizer = layers.TextVectorization(max_tokens=100, output_sequence_length=12)
    
    # 2. ADAPT: Il vectorizer legge tutti i testi per imparare quali parole esistono.
    # Costruisce internamente la mappa: "gatto" -> 5, "cielo" -> 12, ecc.
    vectorizer.adapt(testi)
    
    # 3. TRASFORMAZIONE: Applichiamo la mappa. Ogni frase diventa una lista di 12 numeri.
    # Gli zeri finali che vedrai sono il "silenzio" (padding) per pareggiare le lunghezze.
    sequenze = vectorizer(testi)
    
    return sequenze, vectorizer

def build_modern_rnn_model(vocab_size):
    """
    FASE 3: COSTRUZIONE DEL CERVELLO (MODELLO).
    Qui definiamo come le informazioni fluiscono attraverso i neuroni.
    """
    # --- [A] PORTA D'INGRESSO (INPUT) ---
    # Specifichiamo che riceveremo sequenze di 12 numeri interi.
    inputs = keras.Input(shape=(12,), name="Input_ID_Parole")

    # --- [B] TRADUTTORE DI CONCETTI (EMBEDDING) ---
    # Trasforma ogni numero (es. 5) in un vettore denso (16 numeri decimali).
    # Parole simili avranno vettori simili nello spazio matematico.
    # mask_zero=True: FONDAMENTALE. Dice alla rete: "Se vedi uno zero, è solo 
    # padding per pareggiare la lunghezza, ignoralo nei calcoli della memoria!"
    x = layers.Embedding(input_dim=vocab_size, output_dim=16, mask_zero=True)(inputs)

    # --- [C] IL CUORE DELLA MEMORIA (LSTM) ---
    # Una SimpleRNN dimentica velocemente l'inizio della frase.
    # La LSTM (Long Short-Term Memory) usa dei "gate" (cancelli) per decidere 
    # cosa ricordare e cosa dimenticare del passato. 
    # 32 sono i "neuroni" o unità di memoria interna.
    x = layers.LSTM(32, name="Memoria_Contestuale")(x)

    # --- [D] IL DECISORE FINALE (DENSE) ---
    # Prende il riassunto fatto dalla LSTM e sputa fuori un unico numero.
    # activation="sigmoid": schiaccia il risultato tra 0 e 1 (probabilità).
    outputs = layers.Dense(1, activation="sigmoid", name="Probabilita_Sentiment")(x)

    # --- [E] ASSEMBLAGGIO ---
    model = keras.Model(inputs=inputs, outputs=outputs, name="Rete_Neurale_Sequenziale")
    
    # --- [F] COMPILAZIONE (STRATEGIA DI STUDIO) ---
    # optimizer="adam": l'algoritmo che corregge gli errori durante lo studio.
    # loss="binary_crossentropy": come calcoliamo quanto la rete ha sbagliato.
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

# ================================================================================
# --- CICLO DI ESECUZIONE (LOGICA FLUSSO LAVORO) ---
# ================================================================================

# 1. Recuperiamo i testi originali
testi_raw, etichette = crea_dataset_didattico()

# 2. Trasformiamo i testi in matrici numeriche leggibili dalla GPU
x_train, vect_layer = preprocessamento_dati(testi_raw)

# 3. Curiosità: vediamo come la macchina ha "mappato" le parole
vocabulario = vect_layer.get_vocabulary()
print(f"\n[INFO] Vocabolario Identificato: {vocabulario}")
print(f"[INFO] Esempio Frase 1 (Vettorizzata con Padding):\n{x_train[0]}")

# 4. Creiamo il cervello basandoci sulla dimensione del vocabolario trovato
model = build_modern_rnn_model(len(vocabulario))

# 5. Visualizziamo la struttura interna: vedrai come il "Param #" (numeri da imparare) 
# si sposta dall'Embedding alla LSTM.
print("\n[MAPPA DEL MODELLO]")
model.summary()

# Ora il modello è pronto per essere addestrato con: model.fit(x_train, etichette, epochs=...)