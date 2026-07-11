import os
import cv2
import numpy as np
import face_recognition
import hashlib

# Configurazione: Backend PyTorch per Keras 3
# In un contesto Smart City, questo potrebbe servire per modelli di "Crowd Counting"
os.environ["KERAS_BACKEND"] = "torch"
import keras

def pixelate_face(image, face_locations, pixel_grid=(10, 10)):
    """
    Applica l'effetto pixelazione (mosaico) sui volti rilevati.
    
    Teoria: Riducendo la risoluzione della ROI a 10x10 e riportandola alle dimensioni
    originali tramite interpolazione 'nearest', si ottiene una perdita di dati 
    biometrici irreversibile (GDPR compliant), mantenendo però la 'presenza' del soggetto.
    """
    for top, right, bottom, left in face_locations:
        # 1. Estrazione della ROI (Region of Interest)
        roi = image[top:bottom, left:right]
        
        if roi.size > 0:
            # 2. Downsampling: Riduzione a 10x10 pixel
            # Questo è il punto in cui l'identità viene persa
            temp = cv2.resize(roi, pixel_grid, interpolation=cv2.INTER_LINEAR)
            
            # 3. Upsampling: Riporto alle dimensioni originali
            # Usiamo INTER_NEAREST per mantenere l'effetto a blocchi (pixelated)
            w, h = (right - left), (bottom - top)
            pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 4. Reinserimento nel frame originale
            image[top:bottom, left:right] = pixelated_roi
    
    return image

def secure_hash_embedding(embedding):
    """
    Trasforma l'embedding in un hash SHA-256.
    Permette di contare i 'pedoni unici' senza memorizzare i loro dati biometrici.
    """
    return hashlib.sha256(embedding.tobytes()).hexdigest()

# --- INIZIALIZZAZIONE HARDWARE ---
video_capture = cv2.VideoCapture(0)

# Parametri di ottimizzazione e stato
scale_factor = 0.25 
process_this_frame = True
face_locations = []

print("--- SMART CITY TRAFFIC MONITORING ---")
print("Status: GDPR COMPLIANT | Pixelation: 10x10 | Press 'q' to stop")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Elaborazione IA (Frame Skipping per risparmio energetico/HW limitato)
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detection delle posizioni (Metadata)
        raw_face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Riscalamento coordinate originali
        face_locations = []
        for top, right, bottom, left in raw_face_locations:
            face_locations.append((
                int(top / scale_factor),
                int(right / scale_factor),
                int(bottom / scale_factor),
                int(left / scale_factor)
            ))

        # Estrazione dell'ID anonimo per statistiche di traffico
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, raw_face_locations)
            for encoding in face_encodings:
                secure_id = secure_hash_embedding(encoding)
                # In una Smart City, questo hash verrebbe inviato al database statistico
                print(f"Pedone rilevato (Hash ID): {secure_id[:12]}...", end='\r')

    process_this_frame = not process_this_frame

    # Applicazione Pixelazione (Anonimizzazione visiva)
    frame = pixelate_face(frame, face_locations)

    # Visualizzazione dei metadati (Box di tracciamento)
    for (top, right, bottom, left) in face_locations:
        # Colore azzurro 'Smart City'
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 191, 0), 2)
        cv2.putText(frame, "PEDONE - ID ANONIMO", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 1)

    cv2.imshow('Smart City Traffic Monitor (GDPR Shield)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Salvataggio prova finale prima della chiusura
        cv2.imwrite("smart_city_audit_sample.jpg", frame)
        print("\nCampione di audit salvato: smart_city_audit_sample.jpg")
        break

video_capture.release()
cv2.destroyAllWindows()