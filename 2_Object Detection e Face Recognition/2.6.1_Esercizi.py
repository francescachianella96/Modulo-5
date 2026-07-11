import os

# --- CONFIGURAZIONE AMBIENTE  ---
# Keras 3 con backend Torch per la massima velocità di inferenza
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import requests
import tempfile
from ultralytics import YOLO

class YOLOVideoTracker:
    """
    Pipeline per il tracking pedonale.  Utilizza l'algoritmo BoT-SORT e filtri per singola classe.
    """
    def __init__(self, model_variant='yolov8n.pt'):
        print(f"Inizializzazione YOLO con Tracking Nativo: {model_variant}...")
        self.model = YOLO(model_variant)

    def download_video(self, url):
        """Scarica il video in un file temporaneo per l'elaborazione OpenCV."""
        print(f"Scaricamento video: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in response.iter_content(chunk_size=8192):
            temp_video.write(chunk)
        temp_video.close()
        return temp_video.name

    def process_video(self, video_url, output_path="output_person_botsort.mp4"):
        video_path = self.download_video(video_url)
        cap = cv2.VideoCapture(video_path)
        
        # Estrazione metadati video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Configurazione Output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("Tracking in corso: Filtro PERSONE e algoritmo BoT-SORT...")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 100:
                break

            # 1. 'classes=[0]': Filtra l'inferenza solo sulla classe 'person' (COCO index 0).
            # 2. 'tracker="botsort.yaml"': Passa da ByteTrack a BoT-SORT per una gestione 
            #    delle occlusioni e del movimento camera più robusta.
            results = self.model.track(
                source=frame, 
                persist=True, 
                conf=0.3, 
                iou=0.5, 
                classes=[0],           # <--- Task 1: Filtro Persone
                tracker="botsort.yaml", # <--- Task 2: BoT-SORT
                verbose=False
            )

            # Renderizziamo solo i box delle persone rilevate e i loro ID
            annotated_frame = results[0].plot()

            # Scrittura del frame nel video finale
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 20 == 0:
                print(f"Processati {frame_count} frame...")

        cap.release()
        out.release()
        os.unlink(video_path)
        print(f"Tracking concluso! Video salvato in: {output_path}")

if __name__ == "__main__":
    video_processor = YOLOVideoTracker()
    
    # URL del video di esempio con persone e veicoli
    VIDEO_URL = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    
    video_processor.process_video(VIDEO_URL)