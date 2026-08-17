import os
import numpy as np

# 1. CONFIGURAZIONE AMBIENTE
# Forza Keras 3 a usare PyTorch e comunica a Transformers di fare lo stesso
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 2. SELEZIONE MODELLO REALISTICO
# Usiamo WikiNeural: è pre-addestrato per il NER e riconosce perfettamente l'italiano
model_name = "Babelscape/wikineural-multilingual-ner"

print(f"--- Caricamento modello NER professionale: {model_name} ---")

# Caricamento Tokenizer e Modello con pesi NER reali
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Creazione Pipeline specifica per PyTorch (pt)
# 'aggregation_strategy=None' ci permette di vedere i singoli tag B- e I- per ogni token
ner_pipeline = pipeline(
    "ner", 
    model=model, 
    tokenizer=tokenizer, 
    framework="pt", 
    aggregation_strategy=None 
)

# ==========================================
# 3. FUNZIONE DI ESTRAZIONE ENTITÀ (Punto 2 della traccia)
# ==========================================
def estrai_entita_complesse(testo: str):
    print(f"\n>>> ANALISI TESTO: {testo}")
    print("-" * 60)
    
    # Eseguiamo l'inferenza
    results = ner_pipeline(testo)
    
    print(f"{'Token (WordPiece)':<20} | {'Tag BIO':<8} | {'Confidenza':<10}")
    print("-" * 60)
    
    for res in results:
        # Pulizia estetica per i token WordPiece (rimuove il carattere di spazio Ġ o _)
        clean_word = res['word'].replace(" ", "")
        print(f"{clean_word:<20} | {res['entity']:<8} | {res['score']:.4f}")

# ==========================================
# 4. ANALISI SPECIFICA WORDPIECE (Punto 3 della traccia)
# ==========================================
def verifica_tag_bi(entita_target: str):
    print(f"\n--- VERIFICA DETTAGLIATA TAG B/I: '{entita_target}' ---")
    
    # Vediamo come il tokenizer spezza la stringa
    tokens_wp = tokenizer.tokenize(entita_target)
    print(f"Scomposizione WordPiece: {tokens_wp}")
    
    # Vediamo i tag assegnati
    risultati = ner_pipeline(entita_target)
    
    for r in risultati:
        tag = r['entity']
        word = r['word']
        desc = "Inizio (Beginning)" if tag.startswith("B-") else "Interno (Inside)"
        print(f"Token: {word:<12} | Tag: {tag:<6} | Significato: {desc}")

# ==========================================
# ESECUZIONE TEST REALE
# ==========================================

# Stringa complessa con leggi e date
stringa_test = (
    "Il 1° gennaio 1948 entrò in vigore la Costituzione della Repubblica Italiana, "
    "sostituendo lo Statuto Albertino firmato nel 1848."
)

# Eseguiamo l'estrazione
estrai_entita_complesse(stringa_test)

# Focus richiesto sulla Costituzione
verifica_tag_bi("Costituzione della Repubblica Italiana")