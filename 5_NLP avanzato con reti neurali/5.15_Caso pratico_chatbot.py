"""
================================================================================
ESEMPIO PRATICO: ARCHITETTURA DI UN CHATBOT CON MEMORIA (BACKEND PYTORCH)
================================================================================
In questo script implementiamo un chatbot utilizzando la libreria 'transformers'
di Hugging Face, che è lo standard industriale nel 2026.

Il codice dimostra tre concetti chiave:
1. TOKENIZZAZIONE: Trasformazione del testo in numeri comprensibili ai neuroni.
2. CONTEXT BUFFER: Come mantenere la "memoria" degli ultimi scambi.
3. INFERENZA CAUSALE: La generazione di nuove parole basata sul contesto passato.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURAZIONE ---
# Specifichiamo che il backend preferito è PyTorch. 
# Molte librerie moderne come Keras 3 leggono questa variabile d'ambiente.
os.environ["KERAS_BACKEND"] = "torch"

def initialize_chatbot():
    """
    Inizializza i componenti core dell'IA.
    
    1. Tokenizer: Colui che traduce le parole in ID numerici.
    2. Model: Il cervello (GPT-2) che processa i numeri e genera altri numeri.
    """
    print("\n[STEP 1]: Risveglio dei neuroni (Caricamento GPT-2 via Hugging Face)...")
    
    # Usiamo 'gpt2', un modello bilanciato per dimostrazioni didattiche.
    model_name = "gpt2"
    
    # Carichiamo il 'traduttore' (Tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Carichiamo il 'cervello' (Model) predisposto per la generazione di testo (CausalLM)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # GPT-2 non ha un pad_token di default. Lo impostiamo uguale all'eos_token (End Of Sentence)
    # per evitare errori durante l'elaborazione di sequenze di lunghezza diversa.
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def chat_loop():
    """
    Gestisce il ciclo interattivo, la memoria a breve termine e la generazione.
    """
    # Fase di setup iniziale
    model, tokenizer = initialize_chatbot()
    
    # BUFFER DI MEMORIA: Una lista che funge da 'coda' per i ricordi.
    history_buffer = [] 
    max_history = 3 # Numero massimo di messaggi da 'ricordare' (Sliding Window)

    print("\n" + "="*50)
    print(" CHATBOT ATTIVO (Versione Windows/Torch) ".center(50, "="))
    print("="*50)

    while True:
        # CATTURA INPUT: Leggiamo cosa scrive l'utente nel terminale
        user_message = input("\n[TU]: ").strip()
        
        # COMANDO DI USCITA: Permette di chiudere il programma in modo pulito
        if user_message.lower() in ["exit", "quit"]: 
            print("[BOT]: Spegnimento moduli... A presto!")
            break
            
        # VALIDAZIONE: Evitiamo di sprecare calcoli per input insignificanti
        if len(user_message) < 3: 
            print("[BOT]: Scrivi qualcosa di più lungo, per favore.")
            continue

        # --- FASE 1: GESTIONE DELLA MEMORIA ---
        # Aggiungiamo il nuovo messaggio alla storia con l'etichetta 'User:'
        history_buffer.append(f"User: {user_message}")
        
        # SE IL BUFFER È PIENO: Rimuoviamo il ricordo più vecchio (indice 0).
        # Questo mantiene il prompt entro i limiti di 'Context Window' del modello.
        if len(history_buffer) > max_history: 
            history_buffer.pop(0)

        # --- FASE 2: COSTRUZIONE DEL PROMPT ---
        # Uniamo i messaggi della storia in un'unica stringa separata da invii
        # Aggiungiamo 'Assistant:' alla fine per 'invitare' il bot a rispondere.
        prompt = "\n".join(history_buffer) + "\nAssistant:"
        
        # --- FASE 3: TOKENIZZAZIONE ---
        # Trasformiamo la stringa di testo in 'Tensor' (vettori matematici) per PyTorch.
        # return_tensors="pt" indica proprio il formato PyTorch.
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # --- FASE 4: GENERAZIONE (INFERENZA) ---
        # 'torch.no_grad()' disabilita i calcoli dei gradienti, rendendo tutto più veloce e leggero.
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=50,       # Quante nuove parole vogliamo generare al massimo
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2   # Blocca la ripetizione fastidiosa di coppie di parole
            )
        
        # --- FASE 5: DECODIFICA E PULIZIA ---
        # Trasformiamo i numeri (ID) generati di nuovo in testo leggibile.
        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Estraggiamo solo l'ultima parte della stringa (quella dopo l'ultima etichetta Assistant:)
        # per evitare di mostrare all'utente tutta la storia precedente o codici tecnici.
        response = full_text.split("Assistant:")[-1].strip().split("\n")[0]

        # OUTPUT E AGGIORNAMENTO
        print(f"[BOT]: {response}")
        
        # Aggiungiamo la risposta del bot alla storia per il prossimo turno
        history_buffer.append(f"Assistant: {response}")

# ESECUZIONE
if __name__ == "__main__":
    chat_loop()

# ==============================================================================
# SPIEGAZIONE  DI QUELLO CHE ACCADE SOTTO TRACCIA
# 1. Il Tokenizer spezza la frase: "Ciao" -> [15496]. Questo numero punta a un 
#    vettore di 768 dimensioni nel modello.
# 2. Il Modello (Transformer) analizza i rapporti tra questi numeri (Attention).
# 3. La tecnica 'Causal LM' (Language Modeling) cerca il numero (parola) che ha la 
#    probabilità statistica più alta di apparire dopo la parola 'Assistant:'.
# 4. Lo Sliding Window impedisce errori di "Out of Memory" (OOM) assicurando 
#    che il prompt non diventi infinitamente lungo.
# ==============================================================================