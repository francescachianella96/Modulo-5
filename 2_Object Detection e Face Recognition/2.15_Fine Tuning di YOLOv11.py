import os
import cv2
import time
import numpy as np
import torch  # Utilizzato per la gestione dell'hardware (GPU/CPU) e il rilevamento dei driver

# --- CONFIGURAZIONE AMBIENTE 2026 ---
# Impostiamo il backend di Keras su "torch" per garantire coerenza con il framework Ultralytics, 
# che utilizza PyTorch come motore sottostante per il deep learning.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from ultralytics import YOLO

def train_custom_phone_model():
    """
    Gestisce l'intero workflow di fine-tuning di YOLO11 su un dataset custom.
    Si occupa di: rilevamento percorsi, selezione hardware, training e recupero pesi.
    """
    
    # 1. RILEVAMENTO PERCORSI (Asset Management)
    # Calcoliamo il percorso assoluto della cartella dello script per evitare errori di 'file not found'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Il file data.yaml contiene i metadati del dataset (percorsi immagini, numero di classi, nomi classi)
    dataset_yaml = os.path.join(script_dir, "data.yaml")

    # 2. SELEZIONE DISPOSITIVO (Hardware Acceleration Logic)
    # Interagiamo con 'torch' per determinare la potenza di calcolo disponibile.
    # Se una GPU NVIDIA è presente e configurata correttamente, viene utilizzata per velocizzare il training.
    if torch.cuda.is_available():
        device_to_use = 0  # Identificativo dell'indice della prima GPU
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[INFO]: Accelerazione hardware rilevata ({gpu_name}).")
    else:
        device_to_use = 'cpu' # Fallback su processore standard se la GPU non è disponibile
        print("\n[INFO]: Nessuna GPU rilevata.")

    # 3. ISTANZIAZIONE DEL MODELLO
    # Carichiamo l'architettura YOLO11 (versione 'nano' per velocità).
    # L'oggetto 'model' è l'interfaccia principale della libreria Ultralytics che incapsula la rete neurale.
    model = YOLO("yolo11n.pt") 

    # Validazione preventiva del file di configurazione
    if not os.path.exists(dataset_yaml):
        print(f"\n[ERRORE]: File '{dataset_yaml}' non trovato.")
        return None

    print(f"\n--- FASE DI FINE-TUNING (ADDESTRAMENTO CUSTOM) ---")

    # 4. TRAINING (Optimization Flow)
    # Chiamata al metodo train(): avvia il processo di apprendimento.
    # - data: specifica dove sono i dati.
    # - epochs: quante volte il modello 'vedrà' l'intero dataset.
    # - labs/imgsz: risoluzione di input delle immagini.
    # - amp: Automatic Mixed Precision (ottimizza l'uso della memoria GPU).
    results = model.train(
        data=dataset_yaml,
        epochs=30,      
        imgsz=640,      
        batch=16,       
        name="phone_detector_2026", # Nome della cartella in 'runs/detect/'
        device=device_to_use,
        amp=True 
    )
    
    # --- RECUPERO DINAMICO DEL PERCORSO ---
    # Ultralytics incrementa automaticamente il nome della cartella se esiste già (es. phone_detector_20262).
    # L'oggetto 'results' restituito dal training contiene 'save_dir', che punta alla cartella esatta appena creata.
    save_dir = results.save_dir
    updated_model_path = os.path.join(save_dir, "weights", "best.pt") # 'best.pt' sono i pesi con accuratezza maggiore
    
    print(f"\n--- TRAINING COMPLETATO ---")
    print(f"Modello specializzato individuato in: {updated_model_path}")
    return updated_model_path

def run_webcam_test(model_path):
    """
    Avvia un'istanza di inferenza (rilevamento) in tempo reale analizzando il flusso video della webcam.
    """
    # Controllo di sicurezza: se il training è fallito, usiamo il modello base pre-addestrato
    if not os.path.exists(model_path):
        print(f"Attenzione: Modello '{model_path}' non trovato. Uso il modello base.")
        model_path = "yolo11n.pt"
        
    # Carichiamo il modello (ora con i pesi specializzati per il rilevamento del telefono)
    model = YOLO(model_path)
    # Definiamo il dispositivo per l'inferenza (predizione)
    device_inf = 'cuda' if torch.cuda.is_available() else 'cpu'
    hw_label = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    # Inizializziamo il modulo di cattura video di OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Webcam non disponibile.")
        return

    print(f"\nAvvio Rilevamento Real-time con: {model_path}")
    
    # LOOP PRINCIPALE DI ELABORAZIONE (Pipeline Frame-by-Frame)
    while True:
        # Leggiamo il singolo frame dalla webcam
        ret, frame = cap.read()
        if not ret: break

        # Eseguiamo la predizione sul frame corrente
        # - conf=0.6: soglia di confidenza (interpreta come 'sicuro di almeno il 60%')
        # - results: lista di oggetti che contengono bounding boxes, probabilità e classi
        results = model.predict(source=frame, conf=0.6, verbose=False, device=device_inf)
        
        # Generiamo un frame annotato (disegna automaticamente i rettangoli intorno agli oggetti rilevati)
        annotated_frame = results[0].plot()

        # UI Overlay (Visualizzazione informazioni diagnostiche a schermo)
        # Scriviamo il tipo di hardware utilizzato (GPU o CPU)
        cv2.putText(annotated_frame, f"HW: {hw_label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Scriviamo il nome del file pesi utilizzato
        cv2.putText(annotated_frame, f"FILE: {os.path.basename(model_path)}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Logica Custom: se viene rilevato almeno un telefono (oggetto nella lista boxes)
        if len(results[0].boxes) > 0:
            cv2.putText(annotated_frame, "PHONE DETECTED", (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostriamo il risultato finale in una finestra pop-up
        cv2.imshow("Custom Phone Detection 2026", annotated_frame)
        
        # Gestione interruzione: se l'utente preme 'q' sulla tastiera, il loop termina
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Pulizia risorse hardware e chiusura finestre
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # ENTRY POINT dello script
    # 1. Avviamo la routine di training e salviamo il percorso del modello risultante
    path_to_best = train_custom_phone_model()
    
    # 2. Se il training si è concluso con successo, avviamo il test sulla webcam
    if path_to_best:
        run_webcam_test(path_to_best)