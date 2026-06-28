import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def apply_teal_orange():
    # --- 1. CARICAMENTO ---
    url = "https://picsum.photos/800"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), 1)
        if img is None: raise ValueError("Impossibile decodificare l'immagine.")
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        return None
    
    # --- 2. CREAZIONE LUT TEAL & ORANGE ---
    # Teoria: separiamo i toni sotto e sopra la metà (128)
    # Canale Blu: aumentiamo nelle ombre (+20), diminuiamo nelle luci (-20)
    lut_b = np.array([np.clip(i + 20 if i < 128 else i - 20, 0, 255) for i in range(256)]).astype(np.uint8)
    # Canale Verde: neutro o leggermente incrementato per il Teal
    lut_g = np.array([i for i in range(256)]).astype(np.uint8) 
    # Canale Rosso: diminuiamo nelle ombre (-20), aumentiamo nelle luci (+20)
    lut_r = np.array([np.clip(i - 20 if i < 128 else i + 20, 0, 255) for i in range(256)]).astype(np.uint8)
    
    # Uniamo le LUT e applichiamo
    lut_merge = cv2.merge([lut_b, lut_g, lut_r])
    img_graded = cv2.LUT(img, lut_merge)
    
    # --- 3. FLARE SINTETICO (LENS FLARE) ---
    flare = np.zeros_like(img)
    # Disegniamo un cerchio luminoso nell'angolo in alto a sinistra
    cv2.circle(flare, (img.shape[1]//4, img.shape[0]//4), 100, (50, 80, 100), -1)
    flare = cv2.GaussianBlur(flare, (101, 101), 0)
    
    # Somma additiva: I_final = I_graded + Flare
    img_final = cv2.add(img_graded, flare)

    # --- 4. STAMPA PRIMA E DOPO ---
    # Convertiamo BGR -> RGB per Matplotlib
    img_orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_final_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 7))

    # Immagine Originale
    plt.subplot(1, 2, 1)
    plt.imshow(img_orig_rgb)
    plt.title("Originale (Senza Grading)")
    plt.axis("off")

    # Immagine con Filtro
    plt.subplot(1, 2, 2)
    plt.imshow(img_final_rgb)
    plt.title("Teal & Orange + Light Flare")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return img_final

apply_teal_orange()