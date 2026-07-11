import os
import cv2
import numpy as np
import face_recognition

# Configurazione: Backend PyTorch per Keras
os.environ["KERAS_BACKEND"] = "torch"
import keras

# --- CONFIGURAZIONE PARAMETRI ---
# La tolleranza definisce quanto il sistema è "severo". 
TOLLERANZA = 0.6 

# --- 1. PREPARAZIONE DATABASE (KNOWLEDGE BASE) ---
known_face_encodings = []
known_face_names = []

# Caricamento di Elon Musk
nome_file_elon = "Elon_Musk.jpg"
if os.path.exists(nome_file_elon):
    print(f"Caricamento immagine di {nome_file_elon}...")
    image_elon = face_recognition.load_image_file(nome_file_elon)
    encoding_elon = face_recognition.face_encodings(image_elon)
    
    if len(encoding_elon) > 0:
        known_face_encodings.append(encoding_elon[0])
        known_face_names.append("Elon Musk")
        print("Elon Musk aggiunto al database con successo.")
    else:
        print(f"Errore: Nessun volto trovato in {nome_file_elon}. Prova con un'altra foto.")
else:
    print(f"Nota: {nome_file_elon} non trovato nella cartella dello script.")

# --- 2. INIZIALIZZAZIONE HARDWARE & CATTURA SELFIE ---
video_capture = cv2.VideoCapture(0)

print("\n--- REGISTRAZIONE SELFIE ---")
print("Inquadrati e premi 's' per scattare il tuo selfie.")
print("Premi 'q' per saltare questa fase.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    cv2.putText(frame, "Premi 's' per il Selfie o 'q' per saltare", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Registrazione Iniziale', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame)
        
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append("Utente (Tu)")
            print("Selfie registrato correttamente!")
            cv2.destroyWindow('Registrazione Iniziale')
            break
        else:
            print("Nessun volto rilevato, assicurati che ci sia luce.")
    elif key == ord('q'):
        cv2.destroyWindow('Registrazione Iniziale')
        break

# --- LOGICA DI OTTIMIZZAZIONE ---
scale_factor = 0.25 
process_this_frame = True  # Flag per il Frame Skipping
face_locations = []        # Inizializziamo le liste per evitare errori al primo frame saltato
face_names = []

print(f"\nAvvio Riconoscimento... Tolleranza: {TOLLERANZA} | Frame Skipping: ATTIVO")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Eseguiamo l'inferenza (pesante) solo ogni 2 frame
    if process_this_frame:
        # --- 3. DOWNSCALING & PRE-PROCESSING ---
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # --- 4. INFERENZA (DETECTION & RECOGNITION) ---
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Sconosciuto"
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] <= TOLLERANZA:
                    name = known_face_names[best_match_index]
            
            face_names.append(name)

    # Invertiamo il flag per il prossimo frame
    process_this_frame = not process_this_frame

    # --- 5. VISUALIZZAZIONE DINAMICA ---
    # Notare che usiamo face_locations e face_names dell'ultimo frame elaborato
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = int(top/scale_factor), int(right/scale_factor), int(bottom/scale_factor), int(left/scale_factor)

        color = (0, 255, 0) if name != "Sconosciuto" else (0, 0, 255)
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Face Recognition Real-time', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()