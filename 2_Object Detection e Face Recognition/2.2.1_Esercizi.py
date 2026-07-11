import cv2
import numpy as np
import requests
from io import BytesIO

def get_crowd_image():
    """
    Scarica un'immagine di una folla ottimizzata per il rilevamento.
    Cerchiamo un'immagine dove i volti siano più distinguibili per le Haar Cascades.
    """
    # Nuovo URL: Immagine di una folla (pubblico a un evento) con molti visi visibili
    url = "https://images.unsplash.com/photo-1761839258045-6ef373ab82a7?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    print(f"Scaricamento immagine della folla da: {url}...")
    
    try:
        # User-Agent per simulare un browser ed evitare blocchi
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Errore durante il download: {e}")
        return None

def analyze_crowd_sensitivity():
    """
    Esercizio: Confronto dell'impatto di minNeighbors sul rilevamento dei volti in una folla.
    """
    # 1. Caricamento del classificatore dei volti (modello classico OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 2. Acquisizione immagine
    original_img = get_crowd_image()
    if original_img is None:
        return

    # Conversione in scala di grigi per l'algoritmo di Viola-Jones
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Definiamo i valori di minNeighbors da testare
    # 1: Rosso (Alta sensibilità, molti falsi positivi)
    # 5: Verde (Bilanciamento standard)
    # 10: Blu (Alta precisione, perde i volti difficili)
    test_params = [
        {"val": 1, "color": (0, 0, 255), "label": "minNeighbors=1 (Instabile)"},
        {"val": 5, "color": (0, 255, 0), "label": "minNeighbors=5 (Bilanciato)"},
        {"val": 10, "color": (255, 0, 0), "label": "minNeighbors=10 (Rigoroso)"}
    ]

    print("\n--- Inizio Test Comparativo sulla Folla ---")

    for param in test_params:
        # Copia dell'immagine per non sovrapporre i disegni dei test precedenti
        display_img = original_img.copy()
        n_val = param["val"]
        color = param["color"]
        label = param["label"]

        # Rilevamento Multi-scala
        # scaleFactor=1.1 (riduzione 10% per step)
        # minSize=(25, 25) ridotto leggermente per catturare anche visi più piccoli
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=n_val, 
            minSize=(25, 25)
        )

        # Disegno dei rettangoli di rilevamento
        for (x, y, w, h) in faces:
            cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
        
        # Testo informativo sovrapposto
        cv2.putText(display_img, f"{label} - Volti: {len(faces)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        print(f"Risultato {label}: rilevati {len(faces)} volti.")

        # Visualizzazione finestra
        cv2.imshow('Analisi Sensibilita Haar Cascade', display_img)
        
        print("Premi un tasto per visualizzare il prossimo parametro...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\nEsercizio concluso.")

if __name__ == "__main__":
    analyze_crowd_sensitivity()