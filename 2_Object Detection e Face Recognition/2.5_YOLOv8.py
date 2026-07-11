import os

# --- CONFIGURAZIONE AMBIENTE 2026 ---
# Impostiamo Keras 3 per usare PyTorch come motore di calcolo.
# Questo garantisce interoperabilità nativa con i modelli YOLOv8 basati su Torch.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

class YOLOAdvancedPipeline:
    """
    Pipeline per la computer vision:
    1. Gestione dinamica di immagini da URL
    2. Inferenza scalabile con YOLOv8
    3. Visualizzazione avanzata dei metadati
    """

    def __init__(self, model_variant='yolov8n.pt'):
        """
        Inizializza il modello YOLO.
        Nel 2026, la variante 'nano' (.pt) è lo standard per l'edge computing.
        """
        print(f"Inizializzazione modello {model_variant}...")
        self.model = YOLO(model_variant)

    def download_image(self, url):
        """
        Scarica un'immagine da internet e la converte in formato gestibile.
        Utilizziamo un User-Agent per evitare blocchi dai server web.
        """
        print(f"Scaricamento immagine da: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Convertiamo il contenuto scaricato in un'immagine PIL
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img

    def run_inference(self, image_source):
        """
        Esegue l'intero workflow: download -> predizione -> parsing.
        """
        # Caricamento immagine (se stringa inizia con http, la scarichiamo)
        if isinstance(image_source, str) and image_source.startswith("http"):
            image = self.download_image(image_source)
        else:
            image = image_source

        print("Esecuzione inferenza con parametri ottimizzati...")
        # 'conf=0.25': Filtro confidenza per eliminare il rumore
        # 'iou=0.7': Soglia NMS per la gestione delle sovrapposizioni
        # 'imgsz=640': Risoluzione standard per bilanciare velocità e mAP
        results = self.model.predict(
            source=image,
            conf=0.25,
            iou=0.7,
            imgsz=640,
            save=False
        )

        # Analizziamo il primo risultato del batch
        result = results[0]
        self._parse_metadata(result)
        self._visualize(result)

        return result

    def _parse_metadata(self, result):
        """Estrae i dati tensoriali e li converte in informazioni leggibili."""
        print(f"\n--- Metadati Rilevazione (Trovati {len(result.boxes)} oggetti) ---")
        
        for box in result.boxes:
            # Coordinate XYXY in pixel assoluti
            coords = box.xyxy[0].tolist()
            # Confidenza (sicurezza del modello)
            conf = float(box.conf[0])
            # Classe rilevata
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            print(f"[{class_name.upper()}] Conf: {conf:.2f} | Box: {[round(c, 1) for c in coords]}")

    def _visualize(self, result):
        """Renderizza i box sull'immagine e mostra il risultato finale."""
        # Il metodo .plot() restituisce l'immagine annotata in formato BGR (OpenCV)
        annotated_img = result.plot()

        # Conversione BGR -> RGB per una corretta visualizzazione con PIL/Matplotlib
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Mostriamo il risultato
        final_view = Image.fromarray(annotated_img_rgb)
        
        # Nota: In ambiente locale questo apre il visualizzatore di sistema
        final_view.show()
        
        # Salvataggio su disco per persistenza
        final_view.save("ultimo_rilevamento.png")
        print("\nVisualizzazione generata e salvata come 'ultimo_rilevamento.png'")

# --- ESECUZIONE ---
if __name__ == "__main__":
    # Inizializziamo la pipeline
    pipeline = YOLOAdvancedPipeline()
    
    # URL di test (Esempio: autobus in ambiente urbano)
    URL_INTERNET = "https://ultralytics.com/images/bus.jpg"
    
    # Lanciamo il processo
    pipeline.run_inference(URL_INTERNET)