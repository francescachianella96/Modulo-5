import os
import cv2
import numpy as np
import time
import torch

# --- CONFIGURAZIONE AMBIENTE ---
# Impostiamo il backend PyTorch per Keras 3
os.environ["KERAS_BACKEND"] = "torch"

import keras
from ultralytics import YOLO
import face_recognition

def register_users():
    """
    Fase iniziale di registrazione: acquisisce nomi e foto biometriche dalla webcam.
    """
    known_encodings = []
    known_names = []
    
    print("\n--- CONFIGURAZIONE SISTEMA DI SICUREZZA  ---")
    try:
        n_volti = int(input("Quanti volti vuoi aggiungere al database autorizzato? "))
    except ValueError:
        print("Input non valido. Procedo con 0 utenti.")
        n_volti = 0

    cap = cv2.VideoCapture(0)
    
    for i in range(n_volti):
        nome = input(f"Inserisci il nome per l'utente {i+1}: ")
        print(f"Inquadra {nome} e premi 'c' per catturare la foto biometrica...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Mostriamo le istruzioni a video
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Registrazione: {nome}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Premi 'c' per scattare", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Registrazione Volti", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Trasformiamo in RGB per face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_frame)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(nome)
                    print(f"Utente {nome} registrato con successo!")
                    break
                else:
                    print("Nessun volto rilevato. Riprova assicurandoti che ci sia buona luce.")
        
    cv2.destroyAllWindows()
    cap.release()
    return known_encodings, known_names

class SmartSurveillance:
    def __init__(self, known_encodings, known_names):
        # Utilizziamo YOLO11 Nano per il tracking real-time (SOTA)
        self.detector = YOLO("yolo11n.pt")
        self.known_encodings = known_encodings
        self.known_names = known_names
        
        # Logica di persistenza per estranei
        self.stranger_timers = {}   # ID -> timestamp della prima rilevazione come estraneo
        self.screenshots_taken = set() # Set per evitare screenshot multipli per lo stesso ID
        
        # Creazione cartella alert se non esiste
        if not os.path.exists("alerts"):
            os.makedirs("alerts")
            print("[INFO] Cartella 'alerts' creata per gli screenshot.")
        
        # Ottimizzazione: Rilevamento hardware
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Monitoraggio attivo su dispositivo: {self.device}")

    def start_monitoring(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0
        recognition_frequency = 5 # Analisi facciale ogni 5 frame per risparmio energetico
        
        # Stato persistente degli ID tracciati (ID -> Nome)
        tracked_identities = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # 1. TRACKING PERSONE CON YOLO11
            results = self.detector.track(frame, persist=True, classes=[0], verbose=False, device=self.device)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # 2. LOGICA DI RICONOSCIMENTO
                    if track_id not in tracked_identities or frame_count % recognition_frequency == 0:
                        h, w = frame.shape[:2]
                        person_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        
                        if person_crop.size > 0:
                            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            face_locs = face_recognition.face_locations(rgb_crop)
                            
                            if face_locs:
                                encodings = face_recognition.face_encodings(rgb_crop, face_locs)
                                if encodings:
                                    matches = face_recognition.compare_faces(self.known_encodings, encodings[0], tolerance=0.5)
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        tracked_identities[track_id] = self.known_names[first_match_index]
                                    else:
                                        tracked_identities[track_id] = "ESTRANEO"

                    # 3. LOGICA DI PERSISTENZA E SCREENSHOT
                    identity = tracked_identities.get(track_id, "Analisi...")
                    
                    if identity == "ESTRANEO":
                        color = (0, 0, 255) # Rosso per pericolo
                        
                        # Avviamo il timer per questo ID se non esiste
                        if track_id not in self.stranger_timers:
                            self.stranger_timers[track_id] = time.time()
                        
                        # Calcolo tempo di permanenza
                        elapsed_time = time.time() - self.stranger_timers[track_id]
                        label = f"ESTRANEO! Tempo: {elapsed_time:.1f}s"
                        thickness = 3
                        
                        # Se l'estraneo rimane per più di 1 secondo e non abbiamo ancora fatto lo screenshot
                        if elapsed_time > 1.0 and track_id not in self.screenshots_taken:
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = f"alerts/ID_{track_id}_{timestamp}.jpg"
                            cv2.imwrite(filename, frame)
                            print(f"[ALERT] Intruso ID {track_id} persistente. Screenshot salvato: {filename}")
                            self.screenshots_taken.add(track_id)
                            
                    elif identity == "Analisi...":
                        color = (255, 255, 0)
                        label = f"ID {track_id}: Analisi in corso..."
                        thickness = 2
                    else:
                        # Reset timer se l'ID viene riconosciuto (es. se YOLO scambia ID o errore temporaneo)
                        self.stranger_timers.pop(track_id, None)
                        color = (0, 255, 0) # Verde per autorizzati
                        label = f"AUTORIZZATO: {identity}"
                        thickness = 2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 380, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Cleanup timer per ID non più presenti nel frame (opzionale per gestire la memoria)
            # In una versione industriale si farebbe un check sugli ID attuali vs quelli nel dizionario
            
            cv2.putText(frame, "SISTEMA SORVEGLIANZA - LIVE", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Smart Surveillance System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1. Registrazione utenti autorizzati
    encodings, names = register_users()
    
    # 2. Avvio sistema di sorveglianza
    if not encodings:
        print("Nota: Nessun utente registrato. Tutti i volti verranno segnati come ESTRANEI.")
        
    surveillance = SmartSurveillance(encodings, names)
    surveillance.start_monitoring()