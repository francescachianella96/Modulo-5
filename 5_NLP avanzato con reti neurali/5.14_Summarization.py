"""
Confronto Summarization: BART vs T5 con Keras 3 e PyTorch Backend (Best Practices 2026)
---------------------------------------------------------------------------------------
In questo script confronteremo due dei pesi massimi dell'NLP per il riassunto astrattivo.
Utilizzeremo l'ecosistema Keras 3 con backend PyTorch per massimizzare la flessibilità 
e le performance su GPU.

Modelli:
1. BART (Facebook/Meta): Architettura encoder-decoder pre-addestrata come denoising autoencoder.
2. T5 (Google): Approccio text-to-text (ogni task è visto come una conversazione stringa-a-stringa).

Valutazione:
Useremo la metrica ROUGE (Recall-Oriented Understudy for Gisting Evaluation) per misurare
la sovrapposizione tra i riassunti generati e la ground truth.
"""

import os

# 1. SETUP BACKEND (Best Practice 2026: Definire il backend prima di importare Keras)
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from rouge_score import rouge_scorer

def summarize_text(text, model_name, device, max_length=150):
    """
    Funzione universale per la summarization utilizzando l'ecosistema Transformers
    integrato con il backend Keras/PyTorch.
    """
    print(f"\n--- Generazione con {model_name} ---")
    
    # Caricamento Tokenizer e Modello
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Best Practice: Gestione specifica per T5 (richiede il prefisso del task)
    if "t5" in model_name.lower():
        text = "summarize: " + text

    # Tokenizzazione (Padding e Truncation automatici)
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True
    ).to(device)

    # Generazione con Beam Search (standard per summarization di qualità)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def evaluate_rouge(generated_summary, reference_summary):
    """
    Calcola i punteggi ROUGE-1, ROUGE-2 e ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    
    # Formattiamo i risultati per una lettura pulita
    formatted_scores = {key: round(value.fmeasure * 100, 2) for key, value in scores.items()}
    return formatted_scores

def main():
    # Setup Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Esecuzione su: {device.upper()}")

    # --- DATI DI ESEMPIO (Test di realtà) ---
    original_text = """
    L'intelligenza artificiale generativa ha subito un'accelerazione senza precedenti nel 2025. 
    L'integrazione di architetture multimodali ha permesso ai modelli non solo di scrivere testi, 
    ma di comprendere video e audio in tempo reale con una precisione sovrumana. 
    Keras 3 è diventato lo standard industriale grazie alla sua capacità di astrarre il framework 
    sottostante, permettendo agli sviluppatori di passare da PyTorch a JAX o TensorFlow con una 
    sola riga di codice. Molti esperti ritengono che questa flessibilità sia la chiave per 
    ridurre i costi di addestramento e migliorare l'efficienza energetica dei data center.
    """
    
    reference_summary = "L'AI generativa nel 2025 è diventata multimodale e precisa. Keras 3 è lo standard grazie alla sua flessibilità tra backend come PyTorch, ottimizzando costi ed energia."

    # --- MODELLI DA CONFRONTARE ---
    models_to_test = {
        "BART": "facebook/bart-large-cnn", # Ottimo per riassunti strutturati
        "T5": "t5-base"                    # Versatile e bilanciato
    }

    results = []

    for name, model_id in models_to_test.items():
        try:
            # Generazione
            summary = summarize_text(original_text, model_id, device)
            
            # Valutazione
            scores = evaluate_rouge(summary, reference_summary)
            
            # Archiviazione
            results.append({
                "Modello": name,
                "Riassunto": summary,
                **scores
            })
            
            print(f"Riassunto: {summary}")
            print(f"Scores: {scores}")
            
        except Exception as e:
            print(f"Errore durante il test di {name}: {e}")

    # --- REPORT FINALE ---
    print("\n" + "="*50)
    print("CONFRONTO FINALE PRESTAZIONI")
    print("="*50)
    df = pd.DataFrame(results)
    # Rimuoviamo il testo del riassunto dalla tabella per chiarezza nel print
    print(df.drop(columns=["Riassunto"]).to_string(index=False))

if __name__ == "__main__":
    main()