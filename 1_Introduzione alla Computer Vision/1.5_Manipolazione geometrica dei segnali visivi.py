import os

# Best Practice 2026: Configurazione del backend Keras 3 prima di ogni altro import.
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import requests
from pathlib import Path

def get_random_online_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Scarica un'immagine casuale da internet (Picsum) e la converte in formato OpenCV (BGR).
    """
    url = f"https://picsum.photos/{width}/{height}"
    print(f"[*] Download immagine casuale da: {url}...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Trasformiamo i byte della risposta in un array NumPy
        image_array = np.frombuffer(response.content, np.uint8)
        
        # Decodifichiamo l'array in un'immagine OpenCV (formato BGR di default)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Errore nella decodifica dell'immagine.")
            
        return img
    except Exception as e:
        print(f"[-] Errore durante il download: {e}")
        return None

def demonstrate_geometric_ops():
    """
    Script didattico per mostrare le operazioni geometriche fondamentali.
    """
    # 0. CARICAMENTO (Dalla rete invece che dal path)
    img = get_random_online_image(800, 600)
    
    if img is None:
        return

    h, w = img.shape[:2]

    # --- 1. RESIZE E INTERPOLAZIONE ---
    # Ridimensionamento professionale: 
    # - INTER_AREA per il downscaling (evita aliasing)
    new_w, new_h = 300, 300
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # --- 2. CROPPING CHIRURGICO (NumPy Slicing) ---
    # Lo slicing è una 'View' (O(1)). Formula: img[y_start:y_end, x_start:x_end]
    cy, cx = h // 2, w // 2
    offset = 150
    roi = img[cy-offset:cy+offset, cx-offset:cx+offset]

    
    center = (roi.shape[1] // 2, roi.shape[0] // 2)
    angle = 45 
    scale = 1.0
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    roi_rotated = cv2.warpAffine(
        roi, M, (roi.shape[1], roi.shape[0]), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(0, 0, 0)
    )

    # --- VISUALIZZAZIONE ---
    cv2.imshow("1. Originale (Online)", img)
    cv2.imshow("2. Resized (Downsampled)", img_resized)
    cv2.imshow("3. Cropped ROI", roi)
    cv2.imshow("4. ROI Ruotata", roi_rotated)
    
    print("[*] Premi un tasto qualsiasi sulle finestre per chiudere.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demonstrate_geometric_ops()