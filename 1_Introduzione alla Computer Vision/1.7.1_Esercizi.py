import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import requests

# Configurazione ambiente: Impostiamo Keras 3 con backend PyTorch
os.environ["KERAS_BACKEND"] = "torch"
import keras

def red_segmentation_pipeline(url: str, output_filename: str = "red_segmented.png"):
    """
    Esegue il download, la doppia segmentazione del rosso e il salvataggio.
    """
    
    # --- 1. CARICAMENTO DA INTERNET ---
    print(f"[*] Download in corso da: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Errore: Impossibile decodificare l'immagine.")
    except Exception as e:
        print(f"[-] Errore durante il recupero: {e}")
        return None

    # --- 2. PREPARAZIONE SPAZI COLORE ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # --- 3. DOPPIA MASCHERA PER IL ROSSO ---
    # Range 1: Inizio dello spettro (Hue da 0 a 10)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

    # Range 2: Fine dello spettro (Hue da 170 a 180)
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([179, 255, 255]) # 179 è il limite massimo in OpenCV HSV
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

    # --- 4. OPERAZIONE LOGICA OR ---
    # Combiniamo le maschere: il pixel passa se appartiene a mask1 OPPURE mask2
    full_mask = cv2.bitwise_or(mask1, mask2)

    # Segmentazione sull'immagine originale
    res_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=full_mask)

    # --- 5. SALVATAGGIO (Ritorno a BGR) ---
    # IMPORTANTE: cv2.imwrite si aspetta BGR. Convertiamo il risultato prima di scrivere.
    res_bgr = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(output_filename, res_bgr)
    
    if success:
        print(f"[+] Immagine segmentata salvata correttamente come: {output_filename}")

    # --- 6. VISUALIZZAZIONE ---
    plt.figure(figsize=(15, 5))
    titles = ['Originale (RGB)', 'Maschera Combinata (OR)', 'Segmentazione Rosso']
    images = [img_rgb, full_mask, res_rgb]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray' if i==1 else None)
        plt.title(titles[i], fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # URL di un'immagine con un oggetto rosso evidente (es: una fragola o un'auto)
    target_url = "https://picsum.photos/id/1080/800/600" # Immagine fragole/rosso
    red_segmentation_pipeline(target_url)