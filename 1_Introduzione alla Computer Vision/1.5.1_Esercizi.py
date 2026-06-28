import os

# Best Practice: Configurazione del backend Keras 3 prima di ogni altro import
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import requests
from pathlib import Path

def get_random_online_image(width: int = 800, height: int = 600) -> np.ndarray:
    """Scarica un'immagine casuale da internet e la converte in formato OpenCV."""
    url = f"https://picsum.photos/{width}/{height}"
    print(f"[*] Download immagine casuale da: {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[-] Errore durante il download: {e}")
        return None

def demonstrate_geometric_ops():
    # 0. CARICAMENTO
    img = get_random_online_image(800, 600)
    if img is None: return

    # --- 1. RESIZE (Larghezza 400, mantenendo Aspect Ratio) ---
    target_w = 400
    h, w = img.shape[:2]
    aspect_ratio = h / w
    target_h = int(target_w * aspect_ratio)
    
    # Utilizziamo INTER_AREA perché stiamo rimpicciolendo (da 800 a 400)
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # --- 2. CROP CENTRALE 200x200 ---
    # Calcoliamo il centro della versione ridimensionata
    rh, rw = img_resized.shape[:2]
    cy, cx = rh // 2, rw // 2
    offset = 100 # Metà di 200
    
    # Slicing NumPy: [y_inizio:y_fine, x_inizio:x_fine]
    crop = img_resized[cy-offset:cy+offset, cx-offset:cx+offset]
    
    # --- 3. ROTAZIONE 45° ORARIA ---
    # Nota: In OpenCV gli angoli positivi sono anti-orari. 
    # Per ruotare in senso ORARIO usiamo un angolo negativo (-45).
    center_rotation = (crop.shape[1] // 2, crop.shape[0] // 2)
    angle = -45 
    scale = 1.0
    
    M = cv2.getRotationMatrix2D(center_rotation, angle, scale)
    
    # Applicazione della trasformazione affine
    crop_rotated = cv2.warpAffine(
        crop, M, (crop.shape[1], crop.shape[0]), 
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0) # Sfondo nero per i bordi vuoti
    )

    # --- VISUALIZZAZIONE PASSAGGI ---
    cv2.imshow("1. Originale", img)
    cv2.imshow("2. Resized (w=400)", img_resized)
    cv2.imshow("3. Central Crop (200x200)", crop)
    cv2.imshow("4. Crop Ruotato (-45 deg)", crop_rotated)
    
    print(f"[*] Geometria verificata:")
    print(f"    - Originale: {w}x{h}")
    print(f"    - Ridimensionata: {target_w}x{target_h}")
    print(f"    - Crop: {crop.shape[1]}x{crop.shape[0]}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demonstrate_geometric_ops()