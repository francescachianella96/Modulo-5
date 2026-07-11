import os
import cv2
import time
import numpy as np

# --- ARCHITETTURA DELLA PIPELINE ---
# Questa pipeline integra tre componenti principali:
# 1. Keras/PyTorch: Gestione del backend computazionale.
# 2. Ultralytics (YOLO): Motore di computer vision per la segmentazione.
# 3. OpenCV: Gestione dell'input video (webcam) e dell'output visuale.

# --- CONFIGURAZIONE BACKEND ---
# Impostiamo Keras 3 per utilizzare PyTorch come motore (backend) di calcolo.
# Questa riga deve essere eseguita PRIMA di importare keras stesso.
# Il backend 'torch' è ottimizzato per l'integrazione con i modelli YOLO di Ultralytics.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from ultralytics import YOLO

def run_webcam_segmentation():
    """
    Gestisce l'intero ciclo di vita della segmentazione video:
    Inizializzazione -> Loop di Cattura -> Inferenza -> Post-processing -> Visualizzazione.
    """
    
    # 1. INIZIALIZZAZIONE DEL MODELLO (Classe YOLO)
    # Creiamo un'istanza della classe YOLO fornita dalla libreria Ultralytics.
    # Il file 'yolo11n-seg.pt' contiene i pesi pre-addestrati e l'architettura della rete.
    # 'yolo11n' sta per 'nano', la versione più leggera e veloce, ideale per il real-time.
    print("Caricamento modello YOLO11-seg...")
    model = YOLO("yolo11n-seg.pt") 

    # 2. CONFIGURAZIONE INPUT VIDEO (OpenCV)
    # La classe cv2.VideoCapture inizializza l'accesso alla risorsa hardware (webcam).
    # L'argomento 0 indica la prima fotocamera rilevata dal sistema.
    cap = cv2.VideoCapture(0)
    
    # Controllo di sicurezza: se la risorsa non è disponibile, interrompiamo l'esecuzione.
    if not cap.isOpened():
        print("Errore: Impossibile accedere alla webcam.")
        return

    print("Inizio streaming... Premi 'q' per uscire.")

    # Variabile per tracciare il tempo dell'ultimo frame elaborato (per calcolo FPS).
    prev_time = 0
    
    # 3. LOOP DI ELABORAZIONE (Data Flow)
    # Questo ciclo infinito cattura ed elabora ogni singolo frame proveniente dalla webcam.
    while True:
        # cap.read() restituisce un booleano (success) e l'immagine catturata (frame).
        # Il 'frame' è una matrice NumPy in formato BGR (Blue-Green-Red).
        success, frame = cap.read()
        if not success:
            break

        # CALCOLO FPS (Frames Per Second)
        # Misuriamo la latenza tra il frame corrente e quello precedente per determinare la fluidità.
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # --- FASE DI INFERENZA ---
        # Il metodo model.predict() è il cuore della pipeline. 
        # Interazione: Passiamo il frame di OpenCV (NumPy) al modello YOLO.
        # Parametri chiave:
        # - source=frame: l'immagine da analizzare.
        # - conf=0.5: Confidence Threshold. Scarta tutte le rilevazioni con probabilità < 50%.
        # - iou=0.45: Intersection-over-Union. Gestisce la sovrapposizione tra maschere/box.
        # - show=False/verbose=False: Gestione del display interno e della console per massime prestazioni.
        results = model.predict(source=frame, conf=0.5, iou=0.45, show=False, verbose=False)

        # 4. POST-PROCESSING E VISUALIZZAZIONE
        # Il metodo predict() restituisce una lista di oggetti 'Results'.
        # Ogni oggetto Results contiene: .boxes (rettangoli), .masks (segmentazione), .probs (classi).
        # Il metodo .plot() disegna automaticamente queste informazioni sopra il frame originale.
        annotated_frame = results[0].plot()

        # Inserimento del testo FPS sul frame finale usando OpenCV.
        # Parametri: (immagine, testo, posizione, font, scala, colore BGR, spessore).
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostriamo l'immagine 'annotata' in una finestra grafica creata da OpenCV.
        cv2.imshow("YOLO11-seg Pipeline", annotated_frame)

        # 5. CONTROLLO USCITA
        # cv2.waitKey(1) attende 1ms l'input da tastiera. 
        # '0xFF == ord('q')' verifica se l'utente ha premuto il tasto 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 6. PULIZIA RISORSE
    # È fondamentale rilasciare la webcam e chiudere le finestre per liberare la memoria di sistema.
    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline chiusa correttamente.")

# --- SEZIONE ENTRY POINT ---
# Il blocco seguente garantisce che la funzione venga eseguita solo se il file
# viene lanciato direttamente (non se importato come modulo).
if __name__ == "__main__":
    run_webcam_segmentation()