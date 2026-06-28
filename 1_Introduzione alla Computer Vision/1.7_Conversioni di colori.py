import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import requests

# Configurazione ambiente 2026: Impostiamo Keras 3 con backend PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras

def color_space_analysis_from_url(url: str):
    """
    Scarica un'immagine da un URL, esegue la segmentazione HSV e 
    converte il risultato in un tensore per il Deep Learning.
    """
    
    # --- 1. CARICAMENTO DA INTERNET ---
    print(f"[*] Download in corso da: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Trasformiamo la risposta in un array di byte e decodifichiamo
        # IMREAD_COLOR forza il caricamento in formato BGR (Blue-Green-Red)
        image_array = np.frombuffer(response.content, np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Errore: Impossibile decodificare l'immagine.")
    except Exception as e:
        print(f"[-] Errore durante il recupero dell'immagine: {e}")
        return None

    # --- 2. CONVERSIONE PER VISUALIZZAZIONE (RGB) ---
    # Swap dei canali per Matplotlib: BGR \rightarrow RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- 3. PASSAGGIO ALLO SPAZIO HSV ---
    # Fondamentale per il tracking. 
    # Teoria: In RGB, un'ombra cambia tutti e 3 i canali contemporaneamente.
    # In HSV, il colore è isolato nel canale 'H'.
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- 4. SEGMENTAZIONE: CREAZIONE DI UNA MASCHERA ---
    # Obiettivo: Isolare oggetti di colore Verde.
    # OpenCV HSV: Hue [0-179], Saturation [0-255], Value [0-255].
    lower_green = np.array([35, 50, 50])   
    upper_green = np.array([85, 255, 255]) 

    # Operazione binaria: M(x,y) = 255 se il pixel è nel range, altrimenti 0.
    mask = cv2.inRange(img_hsv, lower_green, upper_green)

    # --- 5. OPERAZIONE BITWISE ---
    # Estraiamo i pixel verdi dall'immagine RGB originale usando la maschera binaria.
    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # --- 6. INTEGRAZIONE KERAS 3 ---
    # Normalizzazione a a[0, 1] e conversione in tensore compatibile con PyTorch.
    tensor_res = keras.ops.convert_to_tensor(res, dtype="float32") / 255.0

    # Visualizzazione didattica
    plt.figure(figsize=(15, 5))
    titles = ['Originale (RGB)', 'Maschera Binaria', 'Segmentazione Verde']
    images = [img_rgb, mask, res]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray' if i==1 else None)
        plt.title(titles[i], fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return tensor_res

# Esecuzione con un'immagine di prova (Paesaggio con molto verde)
if __name__ == "__main__":
    target_url = "https://picsum.photos/id/191/800/600" # Foto di natura
    color_space_analysis_from_url(target_url)