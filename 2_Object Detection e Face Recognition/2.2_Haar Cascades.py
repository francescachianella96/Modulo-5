import cv2
import numpy as np
import requests
from io import BytesIO

def get_random_image_from_web():
    """
    Scarica un'immagine di esempio da Unsplash per testare il rilevatore.
    Unsplash fornisce immagini ad alta risoluzione ottime per la Computer Vision.
    """
    # Utilizziamo un'immagine con un volto ben visibile per il test
    url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?fit=crop&w=800&q=80"
    print(f"Recupero immagine di test da: {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Conversione dei byte ricevuti in un formato leggibile da OpenCV
        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Errore durante il download dell'immagine: {e}")
        return None

def detect_faces_haar_logic():
    """
    Focus: Computer Vision Classica con Haar Cascades.
    Questa funzione dimostra come OpenCV utilizza i file XML pre-addestrati 
    per identificare geometrie facciali senza l'uso di reti neurali profonde.
    """
    
    # --- 1. CARICAMENTO DEI CLASSIFICATORI (LOGICA A CASCATA) ---
    # Le Haar Cascades sono basate su "classificatori deboli" che, messi in cascata,
    # diventano un "classificatore forte". Ogni step scarta le zone che non sembrano un volto.
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # --- 2. ACQUISIZIONE IMMAGINE ---
    img = get_random_image_from_web()
    if img is None:
        return

    # Teoria: Le Haar Cascades operano sulla luminanza (intensità del grigio).
    # Convertiamo l'immagine per ridurre il carico computazionale (da 3 canali a 1).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 3. RILEVAMENTO MULTI-SCALA (THE CORE) ---
    # detectMultiScale crea una piramide di immagini per trovare volti di diverse dimensioni.
    # - scaleFactor=1.1: riduce l'immagine del 10% ad ogni passo per cercare volti più piccoli.
    # - minNeighbors=5: definisce quanti rettangoli "vicini" devono confermare il volto.
    #   Aumentando questo valore si riducono i falsi positivi (ma si rischia di perdere volti reali).
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )

    print(f"Analisi completata. Trovati {len(faces)} potenziali volti.")

    # --- 4. DISEGNO DEI RISULTATI ---
    for (x, y, w, h) in faces:
        # Disegniamo il rettangolo del Volto (Blu)
        # BGR: (255, 0, 0) è Blu in OpenCV, spessore 3
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        # Inseriamo un'etichetta di testo sopra il rettangolo
        cv2.putText(img, 'Volto Rilevato', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # OTTIMIZZAZIONE ROI: Cerchiamo gli occhi SOLO all'interno del volto trovato.
        # Questo riduce drasticamente i falsi positivi (es. bottoni che sembrano occhi).
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Disegniamo il rettangolo degli Occhi (Verde)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # --- 5. MOSTRA OUTPUT ---
    cv2.imshow('Focus: Haar Cascade Detection', img)
    
    print("\nVisualizzazione attiva.")
    print("- Rettangolo BLU: Volto")
    print("- Rettangolo VERDE: Occhi")
    print("\nPremi un tasto qualsiasi sulla finestra dell'immagine per chiudere.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces_haar_logic()