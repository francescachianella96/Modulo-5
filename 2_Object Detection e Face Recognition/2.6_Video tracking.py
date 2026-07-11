import os

"""
Pipeline di Video Tracking con YOLOv8/YOLOv11.

Questo script implementa un flusso di lavoro completo per:
1. Scaricare un video da un URL remoto in un file temporaneo.
2. Inizializzare un modello YOLO con backend ottimizzato.
3. Eseguire il tracking degli oggetti (Object Tracking) con identificativi persistenti.
4. Salvare il risultato elaborato in un nuovo file video.

Dipendenze: ultralytics, opencv-python, requests
"""

# --- CONFIGURAZIONE AMBIENTE ---
# Impostiamo il backend di Keras a "torch" (PyTorch). 
# YOLO di Ultralytics usa internamente PyTorch; forzare l'ambiente aiuta la coerenza
# delle prestazioni e l'allocazione della memoria GPU/CPU in ambienti misti.
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import requests
import tempfile
from ultralytics import YOLO

class YOLOVideoTracker:
    """
    Gestore del tracking video che incapsula la logica di YOLO.
    
    Utilizza il sistema di tracking nativo di Ultralytics, che integra
    algoritmi come ByteTrack e BoT-SORT per mantenere l'identità degli
    oggetti (ID) attraverso i frame.
    """
    def __init__(self, model_variant='yolov8n.pt'):
        """
        Inizializza il modello YOLO.
        :param model_variant: Nome del file del modello (es. 'yolov8n.pt' per nano, 'yolov8s.pt' per small).
        """
        print(f"Inizializzazione YOLO con Tracking Nativo: {model_variant}...")
        # Carica il modello pre-addestrato. Se model_preset esiste nel contesto locale (es. da script esterni), lo usa.
        self.model = YOLO(model_variant)

    def download_video(self, url):
        """
        Scarica un video da un URL e lo salva in un file temporaneo.
        Metodo necessario perché OpenCV (cv2.VideoCapture) legge meglio da file locali che da stream HTTP diretti.
        
        :param url: Link diretto al file video.
        :return: Path del file temporaneo creato.
        """
        print(f"Scaricamento video: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'} # Header per evitare blocchi da parte del server
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status() # Genera un errore se il download fallisce
        
        # Creiamo un file temporaneo che verrà rimosso alla fine dell'elaborazione
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in response.iter_content(chunk_size=8192):
            temp_video.write(chunk)
        temp_video.close()
        return temp_video.name

    def process_video(self, video_url, output_path="output_simple_track.mp4"):
        """
        Esegue l'intera pipeline di processing sul video.
        
        :param video_url: URL di origine.
        :param output_path: Nome del file video prodotto.
        """
        video_path = self.download_video(video_url)
        cap = cv2.VideoCapture(video_path)
        
        # Estrazione metadati necessari per configurare il file di output (stesse dimensioni e velocità del sorgente)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Configurazione del VideoWriter: 'mp4v' è il codec standard per i file .mp4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Elaborazione video con Tracking nativo (ByteTrack)...")
        frame_count = 0

        while cap.isOpened():
            # Legge il frame successivo
            ret, frame = cap.read()
            # Se il video finisce o superiamo i 100 frame (per test rapido), ci fermiamo
            if not ret or frame_count > 100: 
                break

            # --- LOGICA DI TRACKING ---
            # .track() è il metodo di Ultralytics che combina rilevamento e associazione temporale.
            # Parametri chiave:
            # - persist=True: Fondamentale. Comunica al modello che il frame fa parte di una sequenza
            #   e deve mantenere gli ID degli oggetti visti in precedenza.
            # - conf=0.3: Soglia di confidenza. Ignora rilevamenti con probabilità inferiore al 30%.
            # - iou=0.5: Intersection Over Union. Gestisce la soppressione dei box sovrapposti.
            # - tracker="bytetrack.yaml": Specifica l'algoritmo di tracking (ByteTrack è ottimo per fluidità).
            results = self.model.track(
                source=frame, 
                persist=True, 
                conf=0.3, 
                iou=0.5, 
                tracker="bytetrack.yaml",
                verbose=False
            )

            # Il metodo .plot() disegna automaticamente i box, le etichette delle classi (es. "person")
            # e l'ID univoco dell'oggetto (es. "1", "2") sul frame.
            annotated_frame = results[0].plot()

            # Aggiunge il frame annotato al video finale
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 20 == 0:
                print(f"Processati {frame_count} frame...")

        # Rilascio delle risorse hardware e chiusura dei file
        cap.release()
        out.release()
        
        # Pulizia: rimuoviamo il video temporaneo scaricato all'inizio
        if os.path.exists(video_path):
            os.unlink(video_path) 
            
        print(f"Processo concluso! Video salvato in: {output_path}")

# --- PUNTO DI INGRESSO (ENTRY POINT) ---
if __name__ == "__main__":
    # Inizializziamo l'elaboratore (usa il modello nano 'yolov8n.pt' per default)
    video_processor = YOLOVideoTracker()
    
    # URL di un video pubblico contenente persone, biciclette e auto per testare il tracking multi-classe
    VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    
    # Avvio del processo
    video_processor.process_video(VIDEO_URL)