import os
import cv2
import time
import numpy as np

# --- CONFIGURAZIONE AMBIENTE  ---
# Impostiamo Keras 3 per usare PyTorch come motore di calcolo.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from ultralytics import YOLO

def run_webcam_segmentation_with_skipping():
    """
    Esegue la segmentazione d'istanza con Intelligent Frame Skipping.
    Ottimizzato per ridurre il carico computazionale mantenendo la fluidità.
    """
    
    # 1. CARICAMENTO MODELLO
    print("Caricamento modello YOLO11-seg...")
    model = YOLO("yolo11n-seg.pt") 

    # 2. INIZIALIZZAZIONE WEBCAM
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile accedere alla webcam.")
        return

    # Parametri per Intelligent Frame Skipping
    skip_rate = 3           # Eseguiamo l'inferenza ogni 3 frame
    frame_count = 0
    
    # Variabili per statistiche e risparmio temporale
    total_inference_time = 0
    inference_count = 0
    prev_time = time.time()

    print(f"Inizio streaming con SKIP RATE: {skip_rate}...")
    print("Premi 'q' per uscire.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Calcolo FPS generali
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # --- LOGICA DI INTELLIGENT FRAME SKIPPING ---
        if frame_count % skip_rate == 0:
            # Fase di Inferenza (Pesante)
            start_inf = time.time()
            results = model.predict(source=frame, conf=0.5, iou=0.45, show=False, verbose=False)
            inf_duration = time.time() - start_inf
            
            total_inference_time += inf_duration
            inference_count += 1
            
            # Rendering dei risultati (Masks, Boxes, Labels)
            annotated_frame = results[0].plot()
            status_text = "INFERENCE ACTIVE"
            color = (0, 255, 0)
        else:
            # Fase di Skip (Leggera)
            # Manteniamo il frame originale ma aggiungiamo un overlay di stato
            annotated_frame = frame.copy()
            status_text = "SKIP - No Detection"
            color = (0, 0, 255)

        # --- VISUALIZZAZIONE E STATISTICHE ---
        # Overlay informazioni a video
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, status_text, (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Calcolo risparmio percentuale (Teoria vs Pratica)
        if inference_count > 0:
            avg_inf_time = total_inference_time / inference_count
            # Tempo stimato se avessimo processato tutti i frame
            estimated_full_load_time = avg_inf_time * frame_count
            # Risparmio = (Tempo risparmiato / Tempo totale stimato) * 100
            # Poiché processiamo 1 frame ogni N, il risparmio teorico tende a (N-1)/N
            saving_pct = (1 - (inference_count / frame_count)) * 100
            
            cv2.putText(annotated_frame, f"Saving: {int(saving_pct)}%", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("YOLO11-seg Intelligent Skipping", annotated_frame)

        # Log periodico nel terminale (ogni 60 frame)
        if frame_count % 60 == 0:
            print(f"Frame totali: {frame_count} | Inferenze eseguite: {inference_count} | Risparmio stimato: {saving_pct:.1f}%")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. CHIUSURA RISORSE
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam chiusa. Analisi terminata.")

if __name__ == "__main__":
    run_webcam_segmentation_with_skipping()