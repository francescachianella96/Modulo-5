"""
================================================================================================
Fine-tuning di BERT per la Classificazione (Best Practices 2026)
================================================================================================
DESCRIZIONE:
Questo scriptmostra come prendere un modello pre-addestrato (BERT) e "specializzarlo" per un compito
di classificazione del testo (Sentiment Analysis).

INTERAZIONI PRINCIPALI:
1. HUGGING FACE (Libreria Transformers): Fornisce il "corpo" del modello (BertModel) e il 
   "traduttore" (Tokenizer) che trasforma le parole in numeri.
2. KERAS 3: Funge da "regista" (API Funzionale). Definiamo come i dati entrano, come passano 
   attraverso BERT e come arrivano al "decisore finale" (Dense Layer).
3. PYTORCH: Funziona come "motore" sotto il cofano, eseguendo i calcoli matematici richiesti 
   da Keras e BERT.
================================================================================================
"""

import os

# --- PASSO 0: CONFIGURAZIONE BACKEND ---
# Nel 2026, Keras 3 permette di scegliere il motore di calcolo. 
# Impostiamo "torch" (PyTorch) perché BERT nasce e performa ottimamente in questo ecosistema.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from transformers import AutoTokenizer, BertModel
import torch
import numpy as np

def get_tokenizer(model_name="bert-base-uncased"):
    """
    Inizializza il 'traduttore' che converte il testo in numeri comprensibili ai neuroni.
    
    PARAMETRI:
        model_name: Il nome del modello BERT standard (es. 'base' non distingue maiuscole).
    
    COM'È FATTO:
        Usa la classe 'AutoTokenizer' che scarica automaticamente le regole di scomposizione 
        delle parole (WordPiece) specifiche per quella versione di BERT.
    """
    print(f"[INFO] Caricamento del tokenizer per {model_name}...")
    return AutoTokenizer.from_pretrained(model_name)

def build_bert_classifier(model_name="bert-base-uncased", num_classes=2):
    """
    Costruisce l'architettura neurale completa: Corpo di BERT + Testa di Classificazione.
    
    Questa funzione usa l'API Funzionale di Keras per creare un "grafo" di calcolo.
    """
    
    # -------------------------------------------------------------------------
    # 1. DEFINIZIONE DEGLI INGRESSI (Input Layer)
    # -------------------------------------------------------------------------
    # BERT non riceve testo, ma due flussi di numeri:
    input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids") # Gli ID delle parole
    attention_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask") # 1 se è parola, 0 se è spazio vuoto (padding)

    # -------------------------------------------------------------------------
    # 2. IL "CORPO" DI BERT (Encoder)
    # -------------------------------------------------------------------------
    # Carichiamo i pesi già addestrati su miliardi di frasi dal web.
    bert_body = BertModel.from_pretrained(model_name)
    
    # Creiamo un layer Keras personalizzato per "incapsulare" BERT.
    # Questo serve perché BERT (PyTorch) deve comunicare correttamente con Keras.
    class BertLayer(keras.layers.Layer):
        def __init__(self, model, **kwargs):
            super().__init__(**kwargs)
            self.bert = model # Il modello Hugging Face viene salvato qui dentro

        def call(self, inputs):
            # Ingressi: [input_ids, attention_mask]
            # Uscita: estraiamo 'pooler_output', ovvero il riassunto del token [CLS] (Slide 11-14)
            outputs = self.bert(input_ids=inputs[0], attention_mask=inputs[1])
            return outputs.pooler_output

    # Applichiamo il layer di BERT ai nostri ingressi
    # cls_representation è un vettore di 768 numeri che "riassume" l'intera frase
    cls_representation = BertLayer(bert_body)([input_ids, attention_mask])

    # -------------------------------------------------------------------------
    # 3. LA "TESTA" DECISIONALE (Classification Head)
    # -------------------------------------------------------------------------
    # Aggiungiamo uno strato di Dropout per evitare che il modello impari a memoria (overfitting)
    x = layers.Dropout(0.1)(cls_representation)
    
    # Lo strato Denso finale trasforma i 768 numeri di BERT nelle probabilità delle nostre classi.
    # Se num_classes=2 (Sentiment), avremo 2 neuroni in uscita.
    output = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    # -------------------------------------------------------------------------
    # 4. ASSEMBLAGGIO E COMPILAZIONE
    # -------------------------------------------------------------------------
    # Uniamo ingressi e uscite in un unico oggetto Modello
    model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # L'ottimizzatore AdamW è fondamentale nel fine-tuning (Slide 10)
    # Usiamo un learning_rate microscopico (2e-5) per non cancellare la memoria di BERT.
    optimizer = keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=0.01)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy", # Ottima per etichette intere (0, 1, 2...)
        metrics=["accuracy"]
    )
    
    return model

def prepare_dummy_data(tokenizer, texts, labels):
    """
    Trasforma le frasi umane in tensori (matrici di numeri) pronti per essere elaborati.
    
    INTERAZIONE:
        Il tokenizer crea 'input_ids' e 'attention_mask' automaticamente.
    """
    # Padding=True: rende tutte le frasi lunghe uguali aggiungendo zeri
    # Truncation=True: taglia le frasi troppo lunghe (oltre i 512 token)
    encodings = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt" # Restituisce tensori PyTorch
    )
    
    # Distribuiamo i dati in un dizionario che Keras capisce al volo
    x = {
        "input_ids": encodings["input_ids"].numpy(),
        "attention_mask": encodings["attention_mask"].numpy()
    }
    y = np.array(labels) # Le etichette (0 o 1) diventano un array NumPy
    
    return x, y

# ================================================================================================
# AVVIO DEL PROCESSO (MAIN)
# ================================================================================================

if __name__ == "__main__":
    print("\n--- INIZIO LEZIONE PRATICA: FINE-TUNING DI BERT ---")

    # 1. SELEZIONE DEL MODELLO
    # 'bert-base-uncased' è la versione standard (12 layer, 768 neuroni per layer).
    CHECKPOINT = "bert-base-uncased"
    
    # 2. PREPARAZIONE TOKENIZER
    # Trasforma "Ciao" -> [101, 2345, 102]
    tokenizer = get_tokenizer(CHECKPOINT)
    
    # 3. CREAZIONE DATASET DI ESEMPIO (Sentiment Analysis Mini)
    # Immaginiamo di voler classificare se un commento è positivo (1) o negativo (0)
    texts_example = [
        "Incredibile! Questa lezione è chiarissima e utilissima.",   # 1 (Positivo)
        "Purtroppo non ho capito nulla, il codice è troppo difficile.", # 0 (Negativo)
        "Il token [CLS] è fondamentale per capire l'intera frase."     # 1 (Positivo)
    ]
    labels_example = [1, 0, 1]
    
    # Trasformiamo i testi in numeri
    x_train, y_train = prepare_dummy_data(tokenizer, texts_example, labels_example)
    
    # 4. COSTRUZIONE DEL MODELLO
    # Qui avviene la magia: carichiamo BERT e ci montiamo sopra la nostra "testa"
    model = build_bert_classifier(CHECKPOINT, num_classes=2)
    
    # Mostriamo a video lo schema del modello (L'architettura funzionale)
    model.summary()

    # 5. TRAINING (Il cuore del Fine-tuning)
    # Iniziamo a regolare i pesi. Con BERT bastano pochissime epoche (Slide 8).
    print("\n[STEP] Avvio dell'addestramento su Keras con backend PyTorch...")
    model.fit(
        x_train, 
        y_train, 
        epochs=3,      # Passiamo sul dataset 3 volte
        batch_size=2   # Elaboriamo 2 frasi alla volta per non saturare la memoria
    )
    
    # 6. TEST DI PREDIZIONE
    # Proviamo con una frase mai vista prima dal modello
    new_comment = ["Questo approccio integrato Keras-BERT è il futuro!"]
    x_test, _ = prepare_dummy_data(tokenizer, new_comment, [0])
    
    print(f"\n[TEST] Analisi della frase: '{new_comment[0]}'")
    prediction = model.predict(x_test)
    
    # Il risultato è una probabilità: [prob_negativo, prob_positivo]
    print(f"Probabilità (Negativo vs Positivo): {prediction[0]}")
    classe_predetta = np.argmax(prediction)
    print(f"Risultato: {'POSITIVO' if classe_predetta == 1 else 'NEGATIVO'}")

    print("\n--- FINE ESEMPIO ---")