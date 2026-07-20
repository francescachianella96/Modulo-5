import os
import numpy as np

# --- 1. GESTIONE ENCODING (Esercizio per lo studente) ---
def string_to_utf8_bytes(text: str):
    """
    Accetta una stringa e restituisce una lista di numeri interi,
    dove ogni numero è il valore decimale del byte UTF-8 corrispondente.
    """
    # Teoria: UTF-8 è un encoding multi-byte. Ogni carattere può occupare da 1 a 4 byte.
    encoded_bytes = text.encode('utf-8')
    return list(encoded_bytes)

# --- 2. VECTOR SPACE MODEL: DISTANZA EUCLIDEA ---
def demonstrate_semantic_distance():
    """
    Dimostra la distanza euclidea tra vettori in uno spazio semantico fittizio.
    Teoria: La distanza euclidea misura la 'distanza fisica' tra due punti in uno spazio N-dimensionale.
    Formula: sqrt(sum((Ai - Bi)^2))
    """
    # Definizione dei vettori fittizi (3D) forniti dalla traccia
    v_re = np.array([1.0, 0.9, 0.1])
    v_uomo = np.array([0.9, 0.8, 0.2])
    v_regina = np.array([0.1, 0.9, 1.0])

    # Calcolo della Distanza Euclidea tramite NumPy (L2 Norm del vettore differenza)
    dist_re_uomo = np.linalg.norm(v_re - v_uomo)
    dist_re_regina = np.linalg.norm(v_re - v_regina)

    print("--- RISULTATI ANALISI VETTORIALE ---")
    print(f"Distanza Euclidea Re-Uomo: {dist_re_uomo:.4f}")
    print(f"Distanza Euclidea Re-Regina: {dist_re_regina:.4f}")

    # Validazione logica
    if dist_re_uomo < dist_re_regina:
        print("\nVerifica Successo: 'Re' è semanticamente più vicino a 'Uomo' che a 'Regina' in questo spazio.")
    else:
        print("\nVerifica Fallita: Controllare le coordinate dei vettori.")


# --- ESECUZIONE TEST ---
if __name__ == "__main__":
    # Test Parte 1: Encoding
    input_str = "Re"
    bytes_list = string_to_utf8_bytes(input_str)
    print(f"Testo: '{input_str}' -> Valori Byte UTF-8: {bytes_list}\n")

    # Test Parte 2: Distanza Vettoriale
    demonstrate_semantic_distance()