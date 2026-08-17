"""
Sperimentazione di Summarization Astrattiva con BART
----------------------------------------------------
Obiettivo:
1. Caricare 'facebook/bart-large-cnn'.
2. Generare un riassunto di un paragrafo di notizie (~200 parole).
3. Confrontare l'approccio astrattivo con una selezione manuale (estrazione).
4. Analizzare l'impatto del parametro 'num_beams' sulla generazione.

Best Practice: Keras 3 con PyTorch Backend.
"""

import os

# 1. SETUP BACKEND (Deve essere fatto prima di importare Keras)
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

def generate_summary(text, model, tokenizer, device, max_length=50, min_length=20, num_beams=4):
    """
    Esegue la summarization astrattiva variando la complessità della ricerca (num_beams).
    """
    # Tokenizzazione
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    ).to(device)

    # Generazione
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
        length_penalty=2.0
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    # Setup Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Esecuzione su: {device.upper()}")

    # 1. IMPLEMENTAZIONE: Caricamento Modello e Tokenizer
    model_name = "facebook/bart-large-cnn"
    print(f"Caricamento del modello {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    # 2. GENERAZIONE: Paragrafo di notizie (~200 parole)
    news_text = """
    Il 2025 è stato un anno di svolta per la fusione nucleare a confinamento magnetico. Diversi consorzi internazionali, 
    tra cui il progetto ITER in Francia e il Commonwealth Fusion Systems negli Stati Uniti, hanno annunciato traguardi 
    tecnici significativi. In particolare, il reattore sperimentale SPARC ha mantenuto un plasma stabile per una durata 
    record, superando le aspettative degli ingegneri. Questo successo suggerisce che la produzione di energia pulita, 
    sicura e praticamente illimitata potrebbe essere più vicina di quanto previsto inizialmente dai modelli teorici 
    del decennio scorso. Tuttavia, rimangono sfide ingegneristiche monumentali, come lo sviluppo di materiali capaci 
    di resistere a flussi neutronici estremi e la gestione del trizio. Gli investimenti privati nel settore sono 
    triplicati, superando i 15 miliardi di dollari complessivi, segno di una fiducia crescente da parte dei mercati 
    finanziari. Governi di tutto il mondo stanno ora rivedendo i loro piani energetici a lungo termine per integrare 
    la fusione come pilastro della decarbonizzazione globale. Mentre la fissione nucleare continua a dividere 
    l'opinione pubblica, la fusione raccoglie un consenso trasversale grazie all'assenza di scorie a vita lunga 
    e al rischio nullo di meltdown. Se i test del 2026 confermeranno questi dati, la prima immissione in rete 
    di elettricità da fusione potrebbe avvenire entro il 2030.
    """

    # 3. ANALISI: Selezione manuale (Approccio Estrattivo)
    manual_extractive_summary = (
        "1. Il 2025 è stato un anno di svolta per la fusione nucleare con traguardi tecnici significativi in Francia e USA.\n"
        "2. Gli investimenti privati sono triplicati superando i 15 miliardi, indicando fiducia nei mercati.\n"
        "3. La fusione è vista come pilastro della decarbonizzazione entro il 2030 per l'assenza di scorie a vita lunga."
    )

    print("\n" + "="*60)
    print("APPROCCIO ESTRATTIVO (Selezione Manuale delle 3 frasi chiave):")
    print(manual_extractive_summary)
    print("="*60)

    # 4. SFIDA: Variare 'num_beams' da 1 a 4
    results = []
    print("\nAvvio generazione astrattiva con diversi valori di num_beams...")

    for beams in range(1, 5):
        summary = generate_summary(
            news_text, 
            model, 
            tokenizer, 
            device, 
            max_length=50, 
            min_length=20, 
            num_beams=beams
        )
        results.append({
            "num_beams": beams,
            "Riassunto": summary
        })
        print(f"Num Beams {beams}: {summary}")

    # --- CONCLUSIONE ---
    print("\n" + "="*60)
    print("ANALISI E OSSERVAZIONI:")
    print("-" * 60)
    print("1. Ricchezza Lessicale: All'aumentare di num_beams, il modello esplora più cammini di probabilità.")
    print("2. Qualità: Con num_beams=1 (Greedy), il riassunto tende ad essere più ripetitivo o banale.")
    print("3. Con num_beams=4, BART produce solitamente frasi più fluide e coese, catturando meglio il nucleo del testo.")
    print("4. Confronto con Estrazione: Mentre l'approccio manuale 'estrae' intere parti, BART 'rielabora' il concetto,")
    print("   dimostrando capacità di sintesi astrattiva (es. parafrasando 'svolta per la fusione' in concetti più densi).")
    print("="*60)

if __name__ == "__main__":
    main()