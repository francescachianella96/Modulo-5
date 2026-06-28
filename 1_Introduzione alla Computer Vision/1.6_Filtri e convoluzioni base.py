import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

# Best Practice 2026: Configurazione ambiente per Deep Learning Preprocessing
os.environ["KERAS_BACKEND"] = "torch"

def load_image_from_url(url: str) -> np.ndarray:
    """
    Scarica un'immagine da internet e la converte in formato BGR per OpenCV.
    """
    print(f"[*] Recupero immagine da: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # Conversione dei byte in array NumPy e decodifica OpenCV
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine.")
        return img
    except Exception as e:
        print(f"[-] Errore critico: {e}")
        return None

def process_vision_filters(image_url: str):
    # 0. CARICAMENTO DA URL
    img_bgr = load_image_from_url(image_url)
    
    if img_bgr is None:
        return

    # Conversione in RGB per Matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- 1. SMOOTHING (SFOCATURA) ---
    # Kernel di Media: assegna peso 1/N a ogni pixel nel vicinato.
    blur_avg = cv2.blur(img_rgb, (5, 5))

    # Blur Gaussiano: usa una distribuzione normale per preservare meglio le strutture.
    blur_gauss = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    # --- 2. SHARPENING (NITIDEZZA) ---
    # Metodo 1: Unsharp Masking
    # Formula: Risultato = Originale + (Originale - Sfocata) 
    gaussian_mask = cv2.GaussianBlur(img_rgb, (9, 9), 10.0)
    # Formula applicata: 1.5 * Originale - 0.5 * Sfocata.
    img_sharp_sub = cv2.addWeighted(img_rgb, 1.5, gaussian_mask, -0.5, 0)

    # Metodo 2: Kernel di Sharpening Diretto (Matrice di Convoluzione)
    kernel_sharp = np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]], dtype=np.float32)
    img_sharp_kern = cv2.filter2D(img_rgb, -1, kernel_sharp)

    # --- 3. KERNEL PERSONALIZZATO (MOTION BLUR) ---
    size = 15
    kernel_motion = np.zeros((size, size))
    np.fill_diagonal(kernel_motion, 1)
    kernel_motion /= size  # Normalizzazione per evitare sovraesposizione
    
    img_motion = cv2.filter2D(img_rgb, -1, kernel_motion)

    # --- VISUALIZZAZIONE ---
    titles = ['Originale', 'Gaussian Blur', 'Unsharp Mask', 'Sharp Kernel', 'Motion Blur']
    images = [img_rgb, blur_gauss, img_sharp_sub, img_sharp_kern, img_motion]

    plt.figure(figsize=(20, 8))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Esempio con un'immagine casuale ad alta risoluzione
    test_url = "https://picsum.photos/1200/800"
    process_vision_filters(test_url)