import os
import cv2
import numpy as np
import face_recognition

# ==============================================================================
# --- [FASE 0]: PREPARAZIONE TECNICA E CONFIGURAZIONE ---
# ==============================================================================

# Settaggio del Backend di Calcolo
# Spiegazione: Keras (libreria di Deep Learning) può usare diversi "motori" (backend).
# Qui forziamo l'uso di PyTorch per garantire coerenza nell'elaborazione dei tensori.
os.environ["KERAS_BACKEND"] = "torch"
import keras

# TOLLERANZA (Threshold di Distanza Biometrica)
# Concetto Chiave: Il riconoscimento facciale non è un confronto "uguale/diverso" binario,
# ma un calcolo di distanza Euclidea tra vettori di 128 numeri (identikit digitali).
# - Se distanza <= TOLLERANZA -> I volti sono considerati della stessa persona.
# - Valore 0.6: Bilanciamento standard tra sicurezza (False Accept) e comodità (False Reject).
# - Ridurre (es. 0.4): Richiede una somiglianza quasi identica (molto sicuro).
# - Aumentare (es. 0.8): Permette riconoscimenti anche con occhiali o barba, ma rischia errori.
TOLLERANZA = 0.6 

# ==============================================================================
# --- [FASE 1]: COSTRUZIONE DEL DATABASE BIOMETRICO (Nomi ed Encoding) ---
# ==============================================================================

# Liste parallele: known_face_encodings[i] appartiene a known_face_names[i]
known_face_encodings = [] # Conterrà array numpy di 128 valori (gli "identikit")
known_face_names = []     # Conterrà le stringhe con i nomi delle persone

# CARICAMENTO DI ESEMPIO: Elon Musk
# Interazione: os.path.exists verifica la presenza del file su disco.
nome_file_elon = "Elon_Musk.jpg"
if os.path.exists(nome_file_elon):
    print(f"Lettura database: {nome_file_elon}...")
    
    # face_recognition.load_image_file: Carica l'immagine in memoria come array NumPy (RGB).
    image_elon = face_recognition.load_image_file(nome_file_elon)
    
    # face_recognition.face_encodings: Passa l'immagine al modello di Deep Learning (ResNet).
    # Il modello rileva il volto e genera un vettore (encoding) che lo rappresenta.
    encoding_elon = face_recognition.face_encodings(image_elon)
    
    if len(encoding_elon) > 0:
        # Salviamo solo il primo volto trovato nell'immagine
        known_face_encodings.append(encoding_elon[0])
        known_face_names.append("Elon Musk")
        print("Elon Musk aggiunto con successo.")
    else:
        print("Avviso: Nessun volto rilevato nella foto di Elon.")
else:
    print(f"Database: {nome_file_elon} assente. Verrà usato solo l'input real-time.")

# ==============================================================================
# --- [FASE 2]: ACQUISIZIONE DINAMICA (REGISTRAZIONE UTENTE) ---
# ==============================================================================

# cv2.VideoCapture(0): Apre la connessione con la periferica hardware (webcam).
# Lo '0' indica la webcam predefinita del sistema.
video_capture = cv2.VideoCapture(0)

print("\n--- FASE DI REGISTRAZIONE ---")
print("Istruzioni: Guardami e premi 's' per salvare la tua impronta facciale.")

while True:
    # .read(): Cattura un istante (frame) dalla webcam. ret è un booleano (successo/errore).
    ret, frame = video_capture.read()
    if not ret: break
    
    # cv2.putText: Disegna testo sopra l'immagine. Parametri: (immagine, testo, posizione, font, scala, colore, spessore).
    cv2.putText(frame, "Premi 's' per registrarti o 'q' per saltare", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # .imshow: Crea una finestra di Windows per mostrare l'immagine catturata.
    cv2.imshow('Registrazione Volto', frame)

    # .waitKey(1): Aspetta 1ms l'input della tastiera. & 0xFF serve per compatibilità sistemi 64bit.
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        # CONVERSIONE SPAZIO COLORE: OpenCV lavora in BGR (Blue-Green-Red), 
        # m face_recognition richiede RGB (Red-Green-Blue).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generiamo l'identikit dell'utente nel momento dello scatto
        encodings = face_recognition.face_encodings(rgb_frame)
        
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append("Utente (Tu)")
            print("Identità registrata correttamente.")
            cv2.destroyWindow('Registrazione Volto') # Chiude la finestra temporanea
            break
        else:
            print("Errore: Volto non rilevato. Riprova con più luce.")
    elif key == ord('q'):
        cv2.destroyWindow('Registrazione Volto')
        break

# OTTIMIZZAZIONE DELLE PERFORMANCE:
# Elaborare immagini HD in tempo reale è pesante. Rimpiccioliamo l'immagine del 75% (scale 0.25).
# Questo riduce drasticamente i pixel da scansionare, mantenendo i dettagli necessari al riconoscimento.
scale_factor = 0.25 

print(f"\nSISTEMA OPERATIVO! Tolleranza attuale: {TOLLERANZA}")

# ==============================================================================
# --- [FASE 3]: CICLO DI ELABORAZIONE E RICONOSCIMENTO ---
# ==============================================================================

while True:
    # 1. ACQUISIZIONE
    ret, frame = video_capture.read()
    if not ret: break

    # 2. PRE-PROCESSAMENTO (RIDIMENSIONAMENTO E COLORE)
    # cv2.resize: Effettua il downscaling per velocizzare le fasi successive.
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    # Conversione BGR -> RGB necessaria per il modello di face_recognition
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 3. DETECTION (LOCALIZZAZIONE DEI VOLTI)
    # Ritorna una lista di tuple [(top, right, bottom, left), ...] per ogni volto trovato.
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # 4. ENCODING (GENERAZIONE IDENTIKIT)
    # Crea il vettore di 128 numeri per ogni volto trovato nelle posizioni identificate.
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    
    # 5. CONFRONTO (RICERCA NEL DATABASE)
    for face_encoding in face_encodings:
        # face_recognition.face_distance: Calcola la distanza tra il volto vito ora e TUTTI quelli nel DB.
        # Restituisce una lista di scarti numerici.
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        
        name = "Sconosciuto"
        
        if len(face_distances) > 0:
            # np.argmin: Funzione NumPy che trova l'indice dell'elemento col valore MINORE.
            # L'indice del valore minore corrisponde alla persona più simile nel nostro database.
            best_match_index = np.argmin(face_distances)
            distanza_minima = face_distances[best_match_index]
            
            # Applichiamo il filtro della TOLLERANZA
            if distanza_minima <= TOLLERANZA:
                name = known_face_names[best_match_index]
        
        face_names.append(name)

    # ==============================================================================
    # --- [FASE 4]: VISUALIZZAZIONE E FEEDBACK GRAFICO ---
    # ==============================================================================
    
    # Usiamo zip per iterare contemporaneamente sulla posizione e sul nome trovato
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        # RIPRISTINO COORDINATE: Poiché avevamo rimpicciolito l'immagine (75% in meno),
        # dobbiamo moltiplicare le coordinate per 4 (1 / 0.25) per disegnare correttamente sul frame originale.
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        # LOGICA CROMATICA: Verde se riconosciuto, Rosso se estraneo.
        color = (0, 255, 0) if name != "Sconosciuto" else (0, 0, 255)
        
        # DISEGNO: Rettangolo attorno al viso e box per il testo
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # TESTO: Scrive il nome della persona
        cv2.putText(frame, name, (left + 6, bottom - 6), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Mostra il risultato finale nel monitor
    cv2.imshow('Face Recognition Real-time', frame)

    # Interrompi il ciclo infinito se l'utente preme il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================================================================
# --- [FASE 5]: RILASCIO RISORSE ---
# ==============================================================================

# Libera la webcam per altre applicazioni e distrugge le finestre grafiche aperte.
video_capture.release()
cv2.destroyAllWindows()