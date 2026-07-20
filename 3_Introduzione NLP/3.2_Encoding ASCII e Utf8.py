import numpy as np

# --- 1. GESTIONE ENCODING (ASCII vs UTF-8) ---
def demonstrate_encoding():
    # Teoria: ASCII usa 7 bit (max 128 caratteri). UTF-8 è a lunghezza variabile (1-4 byte).
    testo = "AI nel 2026 😛" # Carattere speciale (Emoji) non presente in ASCII
    
    # Codifica in byte (UTF-8 è lo standard universale)
    utf8_encoded = testo.encode('utf-8')
    print(f"Testo Originale: {testo}")
    print(f"Byte UTF-8: {list(utf8_encoded)} (Lunghezza: {len(utf8_encoded)} byte)")
    
    try:
        # Questo fallirà perché l'emoji non esiste nello standard ASCII
        ascii_encoded = testo.encode('ascii')
    except UnicodeEncodeError:
        print("Errore: ASCII non può rappresentare i simboli moderni.")

# --- 2. VECTOR SPACE MODEL: SIMILARITÀ DI COSENO ---
def calculate_similarity(vec_a, vec_b):
    """
    Calcola la similarità di coseno tra due vettori.
    Teoria: Misura l'angolo tra due vettori. Se l'angolo è 0°, il coseno è 1 (massima somiglianza).
    Formula: (A · B) / (||A|| * ||B||)
    """
    # Prodotto scalare (Somma dei prodotti delle componenti corrispondenti)
    dot_product = np.dot(vec_a, vec_b)
    
    # Norma Euclidea (La 'lunghezza' del vettore nello spazio)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # Calcolo della similarità
    return dot_product / (norm_a * norm_b)

# Esecuzione dimostrativa
demonstrate_encoding()

# Simulazione di due parole nello spazio 3D: 'Cane' e 'Gatto'
# In un vero modello, queste coordinate sarebbero apprese dai dati
v_cane = np.array([0.9, 0.1, 0.05])
v_gatto = np.array([0.85, 0.12, 0.07])
v_pietra = np.array([0.1, 0.8, 0.9])

sim_animali = calculate_similarity(v_cane, v_gatto)
sim_pietra = calculate_similarity(v_cane, v_pietra)

print(f"\nSimilarità Cane-Gatto: {sim_animali:.4f} (Vicini)")
print(f"Similarità Cane-Pietra: {sim_pietra:.4f} (Lontani)")