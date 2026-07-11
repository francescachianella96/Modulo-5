import os
import cv2
import numpy as np
import face_recognition
import hashlib

# Configurazione 2026: Backend PyTorch per Keras 3
# Utilizzato per audit di fairness o analisi della qualità dell'immagine
os.environ["KERAS_BACKEND"] = "torch"
import keras

def anonymize_face(image, face_locations, kernel_size=(51, 51)):
    """
    Applica un filtro Gaussian Blur sulle coordinate dei volti rilevati.
    
    Teoria: La sfocatura distrugge le alte frequenze spaziali (dettagli),
    rendendo il dato non identificabile ai sensi del GDPR.
    """
    for top, right, bottom, left in face_locations:
        # Estrazione della Regione di Interesse (ROI)
        roi = image[top:bottom, left:right]
        
        # Applicazione della sfocatura se la ROI è valida
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
            # Reinserimento nel frame originale
            image[top:bottom, left:right] = blurred_roi
    
    return image

def secure_hash_embedding(embedding):
    """
    Genera un hash SHA-256 da un embedding biometrico.
    Assicura che la 'firma' del volto sia conservata in modo non invertibile.
    """
    return hashlib.sha256(embedding.tobytes()).hexdigest()

# --- INIZIALIZZAZIONE HARDWARE ---
video_capture = cv2.VideoCapture(0)

# Parametri di ottimizzazione
scale_factor = 0.25 
process_this_frame = True
face_locations = []

print("Avvio sistema di Privacy Real-time... (Premi 'q' per uscire)")

while True:
    # 1. Cattura frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # 2. Elaborazione IA (ogni 2 frame per performance)
    if process_this_frame:
        # Ridimensionamento per accelerare la detection
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Trova le coordinate dei volti
        # Nota: le coordinate sono relative al frame ridotto
        raw_face_locations = face_recognition.face_locations(rgb_small_frame)
        
        # Riporta le coordinate alla scala originale
        face_locations = []
        for top, right, bottom, left in raw_face_locations:
            face_locations.append((
                int(top / scale_factor),
                int(right / scale_factor),
                int(bottom / scale_factor),
                int(left / scale_factor)
            ))

        # Esempio di hashing (solo se viene rilevato un volto)
        # In un sistema reale, questo verrebbe salvato in un log di audit
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, raw_face_locations)
            for encoding in face_encodings:
                secure_id = secure_hash_embedding(encoding)
                # Stampiamo solo i primi 10 caratteri per brevità
                print(f"Rilevato ID Anonimo: {secure_id[:10]}...", end='\r')

    # Alterna il flag di elaborazione
    process_this_frame = not process_this_frame

    # 3. Applicazione Anonimizzazione (sempre attiva sul frame corrente)
    # Usiamo le ultime face_locations note per evitare sfarfallii
    frame = anonymize_face(frame, face_locations)

    # 4. Visualizzazione Risultato
    for (top, right, bottom, left) in face_locations:
        # Disegniamo un rettangolo sottile per indicare dove sta agendo il filtro
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
        cv2.putText(frame, "ANONIMIZZATO", (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Privacy Shield 2026 - Real-time Anonymization', frame)

    # Uscita
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Pulizia
print("\nChiusura sistema in corso...")
video_capture.release()
cv2.destroyAllWindows()