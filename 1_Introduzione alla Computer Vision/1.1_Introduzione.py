import os

# --- CONFIGURAZIONE ENVIRONMENT 2026 ---
# Keras 3 agnostico con backend PyTorch per massime performance e interoperabilità.
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

def visual_analysis_comprehensive():
    """
    Analisi dei componenti fondamentali di un'immagine.
    Visualizza: Originale, Canali RGB, Struttura (Low Freq) e Dettagli (High Freq).
    """
    
    # --- 1. ACQUISIZIONE DATI ---
    url = "https://picsum.photos/800/600"
    try:
        response = requests.get(url)
        img_rgb = np.array(Image.open(BytesIO(response.content)))
    except Exception as e:
        print(f"Errore download: {e}")
        return

    # --- 2. SCOMPOSIZIONE CROMATICA ---
    # Isoliamo i canali azzerando le altre componenti. 
    # Teoria: un tensore RGB è la sovrapposizione di 3 matrici di intensità.
    red_img = img_rgb.copy(); red_img[:, :, [1, 2]] = 0
    green_img = img_rgb.copy(); green_img[:, :, [0, 2]] = 0
    blue_img = img_rgb.copy(); blue_img[:, :, [0, 1]] = 0

    # --- 3. ANALISI DELLE FREQUENZE SPAZIALI (FFT) ---
    # Convertiamo in scala di grigi: ci interessa la variazione di intensità luminosa.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2 
    
    # Creazione maschere circolari (Filtri Ideali)
    mask_low = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask_low, (ccol, crow), 40, 1, -1) # Raggio 40 per isolare la struttura
    mask_high = 1 - mask_low

    # Ricostruzione nel dominio spaziale tramite IFFT (Inverse FFT)
    # Teoria: Torniamo dai coefficienti di Fourier ai pixel.
    img_low = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * mask_low)))
    img_high = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * mask_high)))

    # --- 4. VISUALIZZAZIONE SCIENTIFICA ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    # Riga 1: Dall'immagine intera ai suoi colori primari
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("1. IMMAGINE ORIGINALE\n(Segnale Completo)", fontweight='bold')
    
    axes[0, 1].imshow(red_img)
    axes[0, 1].set_title("2. CANALE ROSSO\n(Alte lunghezze d'onda)")
    
    axes[0, 2].imshow(green_img)
    axes[0, 2].set_title("3. CANALE VERDE\n(Dettaglio Luminanza)")

    # Riga 2: Dalle frequenze blu alle componenti strutturali
    axes[1, 0].imshow(blue_img)
    axes[1, 0].set_title("4. CANALE BLU\n(Alte energie)")

    axes[1, 1].imshow(img_low, cmap='gray')
    axes[1, 1].set_title("5. BASSE FREQUENZE\n(Struttura e Forme - 'The Blur')")

    axes[1, 2].imshow(img_high, cmap='gray')
    axes[1, 2].set_title("6. ALTE FREQUENZE\n(Bordi e Rumore - 'The Edges')")

    # Pulizia estetica
    for ax in axes.ravel():
        ax.axis('off')

    plt.suptitle("Anatomia di un'Immagine Digitale: Colori vs Frequenze", fontsize=16, y=0.95)
    plt.show()

if __name__ == "__main__":
    visual_analysis_comprehensive()