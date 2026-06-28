import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
from pathlib import Path

# Configurazione Best Practice 2026: Integrazione Keras 3 con Backend PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras

def analyze_edges_from_url(url: str):
    """
    Scarica un'immagine da internet ed esegue l'analisi dei bordi (Sobel e Canny)
    con esportazione finale in un tensore Keras.
    """
    # 1. CARICAMENTO DA INTERNET
    print(f"[*] Download immagine da: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Trasformiamo i byte della risposta in un array NumPy
        image_array = np.frombuffer(response.content, np.uint8)
        
        # Decodifichiamo direttamente in scala di grigi (IMREAD_GRAYSCALE)
        img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine.")
    except Exception as e:
        print(f"[-] Errore durante il caricamento: {e}")
        return None

    # 2. CALCOLO DEI GRADIENTI (SOBEL)
    # Calcoliamo le derivate parziali orizzontali e verticali
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitudo del gradiente (Ipotenusa):
    # G = \sqrt{G_x^2 + G_y^2}
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))

    # 3. ALGORITMO DI CANNY (LOGICA MULTI-STADIO)
    # 
    canny_standard = cv2.Canny(img, 100, 200)

    # 4. TUNING AUTOMATICO (AUTO-CANNY)
    # Utilizziamo la mediana per adattare le soglie dinamicamente
    v = np.median(img)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    canny_auto = cv2.Canny(img, lower, upper)

    # 5. ESPORTAZIONE VERSO DEEP LEARNING (KERAS 3)
    # Conversione in tensore (H, W, 1) per modelli PyTorch/TensorFlow
    edge_tensor = keras.ops.convert_to_tensor(canny_auto, dtype="float32")
    edge_tensor = keras.ops.expand_dims(edge_tensor, axis=-1)

    # Visualizzazione
    plt.figure(figsize=(16, 8))
    images = [img, gradient_magnitude, canny_standard, canny_auto]
    titles = ['Originale (Grayscale)', 'Sobel Gradient', 'Canny (Standard)', 'Canny (Auto-Tuned)']

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return edge_tensor

if __name__ == "__main__":
    # Esempio con un'immagine di architettura (perfetta per gli spigoli)
    urban_url = "https://picsum.photos/id/1031/800/600"
    analyze_edges_from_url(urban_url)