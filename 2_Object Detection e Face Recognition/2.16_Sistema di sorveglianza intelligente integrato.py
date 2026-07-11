"""
================================================================================
AI-GUARD: PIPELINE DI SORVEGLIANZA INTELLIGENTE (SOTA 2026)
================================================================================
Questo sistema implementa una pipeline di sorveglianza multi-livello che combina:
1. Object Tracking (YOLOv11n): Identificazione e persistenza spaziale degli individui.
2. Biometric Recognition (face_recognition): Identificazione univoca dei volti.
3. State Management: Gestione efficiente delle identità per evitare calcoli ridondanti.

TECNOLOGIE UTILIZZATE:
- Ultralytics YOLOv11: State-of-the-art per il tracking real-time.
- Keras 3 & PyTorch: Backend per l'accelerazione hardware.
- Dlib-based Face Recognition: Per l'encoding biometrico ad alta precisione.
================================================================================
"""

import os
import cv2
import numpy as np
import time
import torch

# --- CONFIGURAZIONE AMBIENTE 2026 ---
# Impostiamo il backend PyTorch per Keras 3 per massimizzare le performance su GPU.
# Questo influenza il modo in cui i tensori vengono gestiti internamente.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from ultralytics import YOLO
import face_recognition

def register_users():
    """
    Fase 1: Acquisizione Biometrica (Enrollment).
    
    Questa funzione gestisce l'interfaccia iniziale per registrare utenti autorizzati.
    Raccoglie le immagini dalla webcam, ne estrae i 'biometric encodings' (vettori di 128 numeri 
    che rappresentano il volto) e li associa a un nome.
    
    Returns:
        tuple: (known_encodings, known_names) - Liste parallele di profili biometrici e nomi.
    """
    known_encodings = []
    known_names = []
    
    print("\n--- CONFIGURAZIONE SISTEMA DI SICUREZZA 2026 ---")
    try:
        n_volti = int(input("Quanti volti vuoi aggiungere al database autorizzato? "))
    except ValueError:
        print("Input non valido. Procedo con 0 utenti.")
        n_volti = 0

    # Inizializzazione della cattura video per la registrazione
    cap = cv2.VideoCapture(0)
    
    for i in range(n_volti):
        nome = input(f"Inserisci il nome per l'utente {i+1}: ")
        print(f"Inquadra {nome} e premi 'c' per catturare la foto biometrica...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Rendering UI per la registrazione
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Registrazione: {nome}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, "Premi 'c' per scattare", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Registrazione Volti", display_frame)
            
            # Alla pressione di 'c', processiamo il frame corrente
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # Trasformiamo da BGR (OpenCV) a RGB (face_recognition)
                # Dlib lavora esclusivamente nello spazio colore RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # face_recognition.face_encodings trasforma il volto in un 'embedding' numerico.
                # È questo vettore che verrà confrontato durante il monitoraggio.
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
    """
    Classe core del sistema di monitoraggio.
    Gestisce l'integrazione fra il detector YOLO (spaziale) e il recognizer Dlib (biometrico).
    """
    def __init__(self, known_encodings, known_names):
        """
        Inizializza i motori di intelligenza artificiale.
        
        Args:
            known_encodings: Database dei profili biometrici autorizzati.
            known_names: Nomi associati ai profili.
        """
        # Caricamento di YOLOv11 in versione Nano per garantire frame-rate elevati (Real-time).
        # YOLO si occupa di *trovare* e *seguire* le persone nell'immagine.
        self.detector = YOLO("yolo11n.pt")
        self.known_encodings = known_encodings
        self.known_names = known_names
        
        # Selezione automatica del dispositivo (GPU CUDA o CPU).
        # Il calcolo su GPU è indispensabile per mantenere il tracking fluido.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Monitoraggio attivo su dispositivo: {self.device}")

    def start_monitoring(self):
        """
        Ciclo principale di sorveglianza.
        Implementa una logica a 'due velocità':
        1. YOLO traccia ogni frame (fluidità).
        2. Il riconoscimento facciale interviene a intervalli o per nuovi ID (efficienza).
        """
        cap = cv2.VideoCapture(0)
        frame_count = 0
        recognition_frequency = 5 # Refresh dell'identità ogni 5 frame per risparmiare risorse.
        
        # tracked_identities: Dizionario { Track_ID: Nome/Stato }
        # Questo mantiene la memoria di chi è chi tra un frame e l'altro.
        tracked_identities = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # --- FASE 1: TRACKING SPAZIALE (YOLO) ---
            # .track() con persist=True mantiene lo stesso ID per la stessa persona
            # anche se questa si muove o viene temporaneamente nascosta.
            # classes=[0] isola solo la classe 'persona'.
            results = self.detector.track(frame, persist=True, classes=[0], verbose=False, device=self.device)
            
            # Verifichiamo se ci sono box rilevati con ID validi
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                # Iteriamo su ogni individuo tracciato
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # --- FASE 2: RICONOSCIMENTO BIOMETRICO (Face Recognition) ---
                    # Eseguiamo il riconoscimento solo se:
                    # a) L'ID è nuovo (prima apparizione).
                    # b) È il momento del refresh periodico (frame_count % recognition_frequency == 0).
                    if track_id not in tracked_identities or frame_count % recognition_frequency == 0:
                        
                        # Definiamo la Region of Interest (ROI) corrispondente alla persona
                        h, w = frame.shape[:2]
                        # Clamp coordinate per evitare errori di slicing fuori dai bordi
                        person_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        
                        if person_crop.size > 0:
                            # Preparazione per Dlib
                            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                            
                            # Cerchiamo i volti solo all'interno del riquadro della persona (più veloce)
                            face_locs = face_recognition.face_locations(rgb_crop)
                            
                            if face_locs:
                                # Calcolo encoding per il volto trovato
                                encodings = face_recognition.face_encodings(rgb_crop, face_locs)
                                if encodings:
                                    # Confronto biometria con i volti autorizzati (distanza euclidea < 0.5)
                                    matches = face_recognition.compare_faces(self.known_encodings, encodings[0], tolerance=0.5)
                                    
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        tracked_identities[track_id] = self.known_names[first_match_index]
                                    else:
                                        # Se c'è un volto ma non corrisponde a nessuno nel DB
                                        tracked_identities[track_id] = "ESTRANEO"
                            else:
                                # Se YOLO vede una persona ma il volto non è ancora chiaro/visibile
                                if track_id not in tracked_identities:
                                    tracked_identities[track_id] = "Analisi..."

                    # --- FASE 3: RENDERING UI E ALERTING ---
                    identity = tracked_identities.get(track_id, "Analisi...")
                    
                    # Logica cromatica dello stato di sicurezza
                    if identity == "ESTRANEO":
                        color = (0, 0, 255) # ROSSO: Pericolo rilevato
                        label = "ATTENZIONE: ACCESSO NON AUTORIZZATO!"
                        thickness = 3
                    elif identity == "Analisi...":
                        color = (255, 255, 0) # GIALLO: Verifica in corso
                        label = f"ID {track_id}: Elaborazione in corso..."
                        thickness = 2
                    else:
                        color = (0, 255, 0) # VERDE: Utente autorizzato
                        label = f"AUTORIZZATO: {identity}"
                        thickness = 2

                    # Disegno del rettangolo attorno alla persona (YOLO integration)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Background per l'etichetta (migliora la leggibilità)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 450, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Overlay informativo fisso (System status)
            cv2.putText(frame, "SISTEMA SORVEGLIANZA AI-GUARD 2026 - LIVE", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Visualizzazione a schermo
            cv2.imshow("Smart Surveillance System", frame)
            
            # Chiusura sicura premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Pulizia risorse hardware
        cap.release()
        cv2.destroyAllWindows()

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Inizializziamo il database biometico tramite webcam
    encodings, names = register_users()
    
    if not encodings:
        print("\n[!] ATTENZIONE: Nessun utente registrato.")
        print("Il sistema opererà in modalità 'Zero Trust', segnalando chiunque come ESTRANEO.")
        
    # Istanziamo la classe controller e avviamo il monitoraggio
    surveillance = SmartSurveillance(encodings, names)
    surveillance.start_monitoring()