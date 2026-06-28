import os
import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from io import BytesIO

# --- CONFIGURAZIONE BACKEND 2026 ---
os.environ["KERAS_BACKEND"] = "torch"

def explore_image_arrays():
    """
    Analisi pratica con download dinamico di un'immagine casuale.
    """
    
    # 1. DOWNLOAD IMMAGINE CASUALE
    print("Scaricamento immagine casuale da internet...")
    url = "https://picsum.photos/600/400" # Servizio per immagini random
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Convertiamo i byte della risposta in un array NumPy che OpenCV può leggere
        file_bytes = np.frombuffer(response.content, np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Errore durante il download: {e}")
        # Fallback: immagine sintetica in caso di mancanza di connessione
        img_bgr = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)

    # Convertiamo in RGB (OpenCV legge in BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"Shape dell'immagine (H, W, C): {img_rgb.shape}")
    print(f"Tipo di dato originale: {img_rgb.dtype}")


    # 2. CONVERSIONE IN SCALA DI GRIGI
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    print(f"\nShape in scala di grigi: {img_gray.shape}") 

    # 3. BINARIZZAZIONE (Soglia = 150)
    # cv2.threshold restituisce due valori: la soglia calcolata (uguale a 150 in questo caso) e l'immagine binarizzata
    soglia = 150
    _, img_bin = cv2.threshold(img_gray, soglia, 255, cv2.THRESH_BINARY)

    # 4. CONVERSIONE IN FLOAT32 E NORMALIZZAZIONE [0, 1]
    img_bin_float = img_bin.astype(np.float32) / 255.0
    print(f"Nuovo tipo di dato: {img_bin_float.dtype}")
    print(f"Valore minimo: {img_bin_float.min()}, Valore massimo: {img_bin_float.max()}")

    # 5. CALCOLO PERCENTUALE PIXEL ATTIVI
    # In un'immagine binarizzata e normalizzata, i pixel attivi (bianchi) valgono 1.0, quelli spenti (neri) valgono 0.0.
    # Il metodo np.mean() su valori 0 e 1 restituisce esattamente la frazione di 1 presenti!
    percentuale_attivi = np.mean(img_bin_float) * 100
    print(f"\nPercentuale di pixel attivi (bianchi): {percentuale_attivi:.2f}%")

    # 6. VISUALIZZAZIONE DEI RISULTATI
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("1. Originale (RGB)")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Scala di Grigi")
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"3. Binaria (Soglia {soglia})")
    plt.imshow(img_bin_float, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Per eseguire la funzione:
explore_image_arrays()