"""
====================================================================================================
CORSO DEEP LEARNING: GUIDA DEFINITIVA AI TRANSFORMERS (HUGGING FACE)
====================================================================================================
INTRODUZIONE:
Utilizziamo la libreria 'transformers' di Hugging Face per dimostrare come i modelli 
allo stato dell'arte (BERT, BART, RoBERTa) possano essere utilizzati senza scrivere 
manualmente migliaia di righe di codice neurale.

INTERAZIONI PRINCIPALI:
1. La classe 'pipeline' agisce come ORCHESTRATORE: coordina un Tokenizer (che trasforma il testo 
   in numeri) e un Model (la rete neurale vera e propria) in un unico flusso continuo.
2. PyTorch funge da 'MOTORE' di calcolo, gestendo i tensori e le operazioni matematiche (come la Softmax).
3. L'utente interagisce solo con l'interfaccia di alto livello, inserendo testo e ricevendo logica.
====================================================================================================
"""

import torch                      # Importiamo PyTorch: il "muscolo" matematico che muove i tensori
import numpy as np                # Importiamo NumPy: per la gestione dei dati numerici standard
from transformers import pipeline # Importiamo Pipeline: la "bacchetta magica" di Hugging Face

def analizza_sentiment_semplice():
    """
    Dalla Slide 3: Dimostrazione del 'Nastro Trasportatore' (Pipeline).
    Questa funzione mostra come automatizzare l'analisi del sentiment in 2 passaggi.
    """
    
    print("\n--- [Esempio 1: La Semplicità delle Pipeline] ---")

    # RIGA 1: Creiamo l'istanza della pipeline. 
    # 'sentiment-analysis' specifica il task, 'model' specifica quale supercar pre-addestrata usare.
    # Aggiungiamo framework="pt" per forzare l'uso di PyTorch ed evitare conflitti con Keras.
    analizzatore = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

    # RIGA 2: Definiamo il testo grezzo ricevuto dall'utente o da un database.
    testo = "This Hugging Face library is absolutely life-changing for developers!"
    
    # RIGA 3: Passiamo il testo alla pipeline. 
    # In questo istante il testo viene spezzettato (tokenizzato), passa nei neuroni e viene classificato.
    # Il risultato è una lista di dizionari, prendiamo il primo [0].
    risultato = analizzatore(testo)[0]

    # STAMPA: Mostriamo l'etichetta (POSITIVE/NEGATIVE) e la sicurezza del modello (SCORE).
    print(f"Testo: {testo}")
    print(f"Etichetta: {risultato['label']} | Confidenza: {risultato['score']:.4f}")



def classificazione_zero_shot():
    """
    Dalla Slide 7: Intuizione Artificiale (Zero-Shot).
    Qui il modello usa la logica pura per classificare testi su argomenti mai studiati.
    """
    
    print("\n--- [Esempio 3: Il Miracolo dello Zero-Shot] ---")

    # RIGA 1: Carichiamo un modello specializzato in NLI (Natural Language Inference).
    # Questa pipeline non cerca parole chiave, ma "ragiona" sulle implicazioni semantiche.
    # Forziamo framework="pt" (PyTorch) per stabilità e prestazioni.
    classificatore = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")

    # RIGA 2: Un testo tecnico complesso.
    sequenza = "Il nuovo telescopio spaziale ha catturato immagini di una galassia a 13 miliardi di anni luce."
    
    # RIGA 3: Definiamo le categorie desiderate (possiamo cambiarle in tempo reale senza ri-addestrare!).
    categorie = ["scienza", "politica", "cucina", "sport"]
    
    # RIGA 4: Esecuzione del "duello logico" (Slide 10).
    # Il modello confronta la premessa (sequenza) con l'ipotesi "Questo testo parla di [categoria]".
    risultato = classificatore(sequenza, candidate_labels=categorie)

    # LOOP: Stampiamo ogni categoria con il relativo punteggio di attinenza logica.
    print(f"Testo: {sequenza}")
    for label, score in zip(risultato['labels'], risultato['scores']):
        print(f"  -> Categoria: {label:10} | Probabilità di attinenza: {score:.4f}")


def modelli_multilingue_allineati():
    """
    Dalla Slide 11-14: Abbattimento delle Frontiere Linguistiche.
    Il modello capisce il significato profondo, indipendentemente dalla lingua usata.
    """
    
    print("\n--- [Esempio 4: Sentiment Multilingue e Code-Switching] ---")

    # RIGA 1: Carichiamo BERT multilingue.
    # Questo gigante ha studiato 100+ lingue simultaneamente, allineando i concetti in uno spazio comune.
    modello_multi = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", framework="pt")

    # LISTA: Testiamo diverse sfumature, incluso il 'Code-Switching' (misto lingua).
    esempi = [
        "Il pranzo era delizioso, tornerò sicuramente!", # Italiano puro
        "The software is buggy and constantly crashes.",  # Inglese puro
        "L'esperienza è stata davvero cool!"              # Mistura Italiano/Inglese (Slide 13)
    ]

    # ESECUZIONE: Cicliamo sugli esempi.
    for es in esempi:
        # La pipeline riconosce la lingua e applica i pesi corretti nello spazio latente.
        res = modello_multi(es)[0]
        # Il task restituisce un rating da 1 a 5 stelle (label).
        print(f"Testo: {es:45} | Valutazione (Rating): {res['label']}")


# PUNTO DI INGRESSO (Main): Dove tutto prende vita.
if __name__ == "__main__":
    analizza_sentiment_semplice()       # Dimostra l'astrazione
    classificazione_zero_shot()         # Dimostra l'intuizione logica
    modelli_multilingue_allineati()      # Dimostra la globalità dei modelli

# ====================================================================================================
# CONCLUSIONI E INTERAZIONI TRA COMPONENTI
# ====================================================================================================
# 1. IL FLUSSO DATI: Testo utente -> Pipeline -> Tokenizer (Numeri) -> Transformer (Tensori) 
#    -> Logit (Punteggi) -> Softmax (Probabilità) -> Risultato Finale.
#
# 2. PERCHÉ HUGGING FACE? Perché ci permette di usare modelli con MILIARDI di parametri 
#    come se fossero semplici funzioni Python, nascondendo la complessità dei gradienti e 
#    delle matrici (Slide 1).
#
# 3. IL RUOLO DI PYTORCH: In questo script, PyTorch è l'infrastruttura silenziosa che 
#    permette ai Transformers di "pensare". Gestisce la memoria della GPU/CPU e calcola 
#    velocemente le funzioni di attivazione come la Softmax.
# ====================================================================================================