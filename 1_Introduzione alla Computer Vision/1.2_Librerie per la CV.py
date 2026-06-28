import os

# --- CONFIGURAZIONE BACKEND 2026 ---
# Keras 3 permette di scegliere il motore computazionale. 
# Scegliamo PyTorch per la sua flessibilità nel Deep Learning moderno.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import cv2  # OpenCV: Il motore per il tempo reale
from PIL import Image  # Pillow: Lo standard per la gestione file e metadati
from skimage import filters, color, util  # scikit-image: Analisi scientifica
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import keras # Utilizzato qui per utilities di pre-processing e salvataggio coerente

def workshop_librerie_immagini():
    """
    Dimostrazione dell'interoperabilità tra librerie.
    Concetto Teorico: L'Immagine come 'Dato Universale' (NumPy ndarray).
    """
    
    # 1. CARICAMENTO CON PILLOW (PIL)
    # Teoria: Pillow è eccellente per il caricamento iniziale e la gestione dei formati web.
    url = "https://picsum.photos/1200/800"
    response = requests.get(url)
    # Carichiamo come oggetto Image di Pillow (formato RGB di default)
    img_pil = Image.open(BytesIO(response.content))
    print(f"Caricata con PIL: {type(img_pil)} | Formato: {img_pil.format}")

    # 2. CONVERSIONE IN NUMPY (Il ponte)
    # Teoria: Le reti neurali leggono array, non oggetti. La conversione è il primo passo del pre-processing.
    img_array = np.array(img_pil)
    print(f"Convertita in NumPy: {img_array.shape} | Dtype: {img_array.dtype}")

    # 3. ELABORAZIONE CON SCIKIT-IMAGE
    # Teoria: scikit-image opera nativamente su array NumPy trattandoli come segnali scientifici.
    # Applichiamo un filtro di Sobel per trovare i bordi (Edge Detection).
    gray_sk = color.rgb2gray(img_array) # Converte in float64 [0, 1]
    edges = filters.sobel(gray_sk)

    # 4. ELABORAZIONE CON OPENCV (Problema BGR)
    # Teoria: OpenCV usa lo spazio colore BGR. Se passiamo un array RGB senza convertirlo, 
    # i canali Rosso e Blu verranno scambiati.
    # Eseguiamo un blur Gaussiano per ridurre il rumore ad alta frequenza.
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    blurred_bgr = cv2.GaussianBlur(img_bgr, (15, 15), 0)
    # Torniamo in RGB per la visualizzazione corretta
    blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)

    # 5. SALVATAGGIO CON KERAS 3 (Best Practice 2026)
    # Teoria: Keras 3 standardizza il salvataggio per garantire che i dati siano 
    # pronti per l'input di un modello.
    keras.utils.save_img("output_elaborato.jpg", blurred_rgb)
    print("Immagine salvata correttamente utilizzando Keras 3 utilities.")

    # 6. VISUALIZZAZIONE SCIENTIFICA CON MATPLOTLIB
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Originale (Pillow -> NumPy)")
    plt.imshow(img_array)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Bordi (scikit-image)")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Sfocatura (OpenCV)")
    plt.imshow(blurred_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    workshop_librerie_immagini()