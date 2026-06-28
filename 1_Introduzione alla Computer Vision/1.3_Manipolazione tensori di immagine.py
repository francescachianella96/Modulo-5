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

    # 2. CONVERSIONE E NORMALIZZAZIONE
    # Teoria: La normalizzazione in [0, 1] previene saturazioni nei calcoli.
    img_float = img_rgb.astype(np.float32) / 255.0
    print(f"Nuovo Dtype: {img_float.dtype} | Range: [{img_float.min():.2f}, {img_float.max():.2f}]")

    # 3. RIDUZIONE DIMENSIONALE (Grayscale)
    # Formula Luminanza: Y = 0.299R + 0.587G + 0.114B
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 4. BINARIZZAZIONE (Thresholding)
    _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 5. VISUALIZZAZIONE COMPARATIVA
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Originale (Random RGB)")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Scala di Grigi")
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Binarizzata (Otsu-style)")
    plt.imshow(img_binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 6. SALVATAGGIO DINAMICO
    # Usiamo il metodo corretto per aggiungere la dimensione del canale (H, W, 1)
    img_to_save = np.expand_dims(img_binary, axis=-1)
    keras.utils.save_img("processed_random_result.png", img_to_save)
    print(f"Risultato salvato come 'processed_random_result.png' con shape {img_to_save.shape}")

if __name__ == "__main__":
    explore_image_arrays()