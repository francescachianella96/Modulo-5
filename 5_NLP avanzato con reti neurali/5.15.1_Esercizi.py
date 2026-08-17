import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Forza l'uso di PyTorch
os.environ["KERAS_BACKEND"] = "torch"

def initialize_chatbot():
    print("\n[STEP 1]: Caricamento GPT-2 (Shakespeare Edition)...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def chat_loop():
    model, tokenizer = initialize_chatbot()
    
    # MODIFICA 1: Memoria estesa a 5 messaggi
    history_buffer = [] 
    max_history = 5 
    
    # MODIFICA 2: Prompt di sistema in inglese per lo stile Shakespeare
    system_instruction = (
        "System: You are William Shakespeare. Respond to the user using "
        "Early Modern English, with 'thou', 'thee', 'thy', and poetic metaphors. "
        "Maintain a dramatic and theatrical tone."
    )

    print("\n" + "="*50)
    print(" CHATBOT SHAKESPEAREANO ATTIVO ".center(50, "="))
    print("="*50)

    while True:
        user_message = input("\n[TU]: ").strip()
        if user_message.lower() in ["exit", "quit"]: break
        if len(user_message) < 2: continue

        # Aggiornamento Memoria
        history_buffer.append(f"User: {user_message}")
        
        # Sliding Window: manteniamo gli ultimi 5 messaggi
        if len(history_buffer) > max_history: 
            history_buffer.pop(0)

        # Costruzione del Prompt con Personaggio e Memoria
        # Inseriamo l'istruzione di sistema in cima per dare il contesto
        prompt = system_instruction + "\n\n" + "\n".join(history_buffer) + "\nAssistant:"
        
        # Generazione
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=60, 
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                do_sample=True, # Abilitiamo il campionamento per risposte più creative
                temperature=0.8 # Leggermente alta per favorire lo stile poetico
            )
        
        # Decode e Pulizia
        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # Estraiamo solo l'ultima risposta dell'assistente
        try:
            response = full_text.split("Assistant:")[-1].strip().split("\n")[0]
        except:
            response = "Alas, my tongue is tied! (Errore nel parsing)"

        print(f"[BOT]: {response}")
        
        # Salviamo la risposta nella memoria per mantenere il filo del discorso
        history_buffer.append(f"Assistant: {response}")

if __name__ == "__main__":
    chat_loop()