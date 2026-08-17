import numpy as np

def softmax(x):
    """Calcola la funzione softmax per ogni punteggio in x."""
    e_x = np.exp(x - np.max(x)) # Sottraiamo il max per stabilità numerica
    return e_x / e_x.sum()

def manual_dot_product_attention(encoder_states, decoder_state):
    """
    Implementazione manuale del meccanismo di Attention tramite prodotto scalare.
    
    Argomenti:
    encoder_states: Array numpy di forma (n_stati, dim_vettore)
    decoder_state: Array numpy di forma (dim_vettore,)
    
    Ritorna:
    context_vector: Il vettore di contesto finale
    weights: I pesi dell'attenzione (somma = 1)
    """
    
    print("--- 1. Calcolo dei Punteggi (Scores) ---")
    # Calcoliamo il prodotto scalare tra lo stato del decoder e ogni stato dell'encoder
    # dot_product(decoder, encoder_i)
    scores = np.dot(encoder_states, decoder_state)
    print(f"Punteggi grezzi (Dot Product): {scores}")
    
    print("\n--- 2. Calcolo dei Pesi (Softmax) ---")
    # Applichiamo Softmax per trasformare i punteggi in probabilità (pesi)
    weights = softmax(scores)
    print(f"Pesi di attenzione (Alpha): {weights}")
    print(f"Verifica somma pesi: {np.sum(weights):.2f}")
    
    print("\n--- 3. Calcolo del Vettore di Contesto ---")
    # Il vettore di contesto è la somma pesata degli stati dell'encoder
    # context = sum(weight_i * encoder_state_i)
    context_vector = np.dot(weights, encoder_states)
    print(f"Vettore di contesto finale: {context_vector}")
    
    return context_vector, weights

# --- ESECUZIONE ---

# 1. Definiamo i parametri
dim_vettore = 4
n_stati_encoder = 3

# 2. Generiamo vettori casuali per simulare gli stati
np.random.seed(42) # Per riproducibilità
encoder_states = np.random.rand(n_stati_encoder, dim_vettore)
decoder_state = np.random.rand(dim_vettore)

print("Stati dell'Encoder (Memoria):")
print(encoder_states)
print("\nStato del Decoder (Query attuale):")
print(decoder_state)
print("-" * 30)

# 3. Applichiamo la funzione
context, weights = manual_dot_product_attention(encoder_states, decoder_state)

# 4. Verifica finale richiesta
assert np.isclose(np.sum(weights), 1.0), "ERRORE: La somma dei pesi non è pari a 1!"
print("\n[OK] Test superato: La somma dei pesi è esattamente 1.0")