import os
import cv2
import numpy as np
import requests
from io import BytesIO
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURAZIONE AMBIENTE 2026 ---
# Impostiamo il backend Torch per Keras 3 per garantire la massima compatibilità con YOLOv8
os.environ["KERAS_BACKEND"] = "torch"

def download_image(url):
    """
    Scarica un'immagine da internet e la converte in formato PIL.
    """
    print(f"Scaricamento immagine da: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    # Convertiamo il contenuto scaricato in un'immagine PIL
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def process_and_filter_people(image_source):
    """
    Carica YOLOv8n, rileva oggetti con confidenza > 0.5 e filtra
    esclusivamente la classe 'person' (ID 0). Gestisce sia URL che percorsi locali.
    """
    
    # 1. CARICAMENTO MODELLO
    print("Inizializzazione modello YOLOv8n...")
    model = YOLO("yolov8n.pt")

    # 2. GESTIONE SORGENTE IMMAGINE
    if isinstance(image_source, str) and image_source.startswith("http"):
        image = download_image(image_source)
    else:
        image = image_source

    # 3. ESECUZIONE INFERENZA FILTRATA
    # Parametri:
    # 'conf=0.5': Filtra a monte tutto ciò che ha sicurezza inferiore al 50%
    # 'classes=[0]': Filtra a monte solo la classe 'person' (ID 0)
    print(f"Elaborazione in corso...")
    results = model.predict(
        source=image,
        conf=0.5,
        classes=[0],  # Filtro per classe 'person'
        verbose=False
    )

    # 4. ANALISI DEI RISULTATI
    result = results[0]
    num_persone = len(result.boxes)
    
    print("-" * 30)
    print(f"RIEPILOGO RILEVAZIONE")
    print(f"Numero totale di persone rilevate (Conf > 0.5): {num_persone}")
    print("-" * 30)

    # 5. VISUALIZZAZIONE
    # .plot() genera l'immagine con i box solo per i risultati filtrati sopra
    annotated_img = result.plot()
    
    # Conversione da BGR (OpenCV) a RGB (PIL/Standard)
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Visualizzazione dell'immagine
    img_output = Image.fromarray(annotated_img_rgb)
    img_output.show()
    
    # Salvataggio opzionale
    save_name = "rilevamento_persone_filtrato.jpg"
    img_output.save(save_name)
    print(f"Risultato salvato come '{save_name}'")

if __name__ == "__main__":
    # URL di test (immagine bus di Ultralytics)
    URL_TEST = "https://ultralytics.com/images/bus.jpg"
    
    try:
        process_and_filter_people(URL_TEST)
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")