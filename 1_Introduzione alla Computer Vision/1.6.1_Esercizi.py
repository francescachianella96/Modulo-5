import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

# Best Practice: Configurazione ambiente per Deep Learning Preprocessing
os.environ["KERAS_BACKEND"] = "torch"

def load_image_from_url(url: str) -> np.ndarray:
    """
    Scarica un'immagine da internet e la converte in formato BGR per OpenCV.
    """
    print(f"[*] Recupero immagine da: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine.")
        return img
    except Exception as e:
        print(f"[-] Errore critico: {e}")
        return None

def process_emboss_filter(image_url: str):
    # 0. CARICAMENTO DA URL
    img_bgr = load_image_from_url(image_url)
    if img_bgr is None:
        return

    # --- 1. PREPARAZIONE: SCALA DI GRIGI ---
    # Il filtro Emboss lavora meglio sui gradienti di intensità luminosa
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. DEFINIZIONE KERNEL EMBOSS ---
    # Un kernel Emboss mette in risalto le differenze di pixel lungo una direzione.
    # Usiamo valori opposti (-2 e 2) per enfatizzare i bordi.
    kernel_emboss = np.array([[-2, -1,  0],
                              [-1,  1,  1],
                              [ 0,  1,  2]], dtype=np.float32)

    # --- 3. APPLICAZIONE FILTRO CON BIAS ---
    # ddepth=-1: mantiene la stessa profondità dell'immagine sorgente.
    # delta=128: aggiunge un valore di "bias" a ogni pixel dopo la convoluzione.
    # Questo sposta lo zero matematico verso il grigio medio (128), 
    # rendendo visibili sia le ombre (valori negativi) che le luci (valori positivi).
    img_emboss = cv2.filter2D(img_gray, -1, kernel_emboss, delta=128)

    # --- VISUALIZZAZIONE ---
    plt.figure(figsize=(15, 7))

    # Immagine Originale (Convertita in RGB per Matplotlib)
    plt.subplot(1, 2, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Originale (Urban Landscape)", fontsize=14, fontweight='bold')
    plt.axis('off')

    # Immagine con Filtro Emboss
    plt.subplot(1, 2, 2)
    plt.imshow(img_emboss, cmap='gray')
    plt.title("Filtro Emboss (Grayscale + 128 Bias)", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Utilizzo un'immagine che tipicamente contiene molte linee geometriche (città)
    test_url = "https://picsum.photos/id/103/1200/800" # Immagine di città/strade
    process_emboss_filter(test_url)