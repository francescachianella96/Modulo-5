"""
Controllo Totale della Generazione GPT-2
-----------------------------------------------------------
Questo script carica GPT-2 e genera testo utilizzando un unico prompt,
configurando e spiegando ogni parametro di campionamento direttamente.
"""

import os

# Configurazione obbligatoria per usare Keras con l'anima di PyTorch
os.environ["KERAS_BACKEND"] = "torch"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def esempio_generazione_semplice():
    """
    Esegue una singola generazione di testo commentando ogni parametro tecnico.
    """
    
    # 1. PREPARAZIONE (Caricamento rapido)
    print("Caricamento modello...")
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Spostamento su GPU se presente per velocizzare (Best Practice)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. DEFINIZIONE DEL PROMPT (Input)
    prompt = "In the year 2050, artificial intelligence will"
    
    # Trasformiamo il testo in numeri (Input IDs)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3. GENERAZIONE CON TUTTI I PARAMETRI (Il cuore della lezione)
    print(f"\nPrompt iniziale: {prompt}")
    print("-" * 30)

    output_tokens = model.generate(
        **inputs,
        
        # --- Parametri di Lunghezza ---
        max_new_tokens=40,       # Quanti nuovi token (parole) generare al massimo
        
        # --- Cuore del Sampling ---
        do_sample=True,          # TRUE: abilita la fantasia (sampling). FALSE: usa la Greedy Search (sempre il più probabile)
        
        # --- Strategie di Strategia ---
        temperature=0.8,         # Controlla la confidenza: <1.0 = conservativo, >1.0 = creativo/caotico
        top_k=50,                # Limita la scelta alle 50 parole più probabili (riduce il rischio di errori gravi)
        top_p=0.92,              # Nucleus Sampling: sceglie tra le parole che sommate arrivano al 92% di probabilità
        
        # --- Gestione della Ripetizione ---
        repetition_penalty=1.2,  # Evita che il modello scriva la stessa parola o frase all'infinito
        
        # --- Configurazione Tecnica ---
        pad_token_id=tokenizer.eos_token_id  # Indica al modello come gestire gli spazi vuoti
    )

    # 4. TRADUZIONE OUTPUT (Da numeri a parole)
    testo_generato = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    print(f"Risultato:\n{testo_generato}")

if __name__ == "__main__":
    esempio_generazione_semplice()