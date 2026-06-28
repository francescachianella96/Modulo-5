import os

# Best practice 2026: Configurazione del backend di Keras 3 prima dell'importazione
# Utilizziamo PyTorch come motore di calcolo per la sua flessibilità nel Deep Learning
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras
from pathlib import Path
from typing import Optional
import requests

def download_demo_images(target_dir: str, count: int = 2):
    """
    Scarica immagini casuali da internet per popolare la cartella di test.
    """
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Download di {count} immagini di test in corso...")
    
    for i in range(count):
        # Utilizziamo Picsum per ottenere immagini casuali ogni volta
        url = f"https://picsum.photos/640/480?random={i}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                file_path = path / f"test_image_{i}.jpg"
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"[+] Scaricata: {file_path.name}")
        except Exception as e:
            print(f"[-] Errore durante il download: {e}")


def process_image_pipeline(image_path: Path, output_folder: Path):
    """
    Esegue il caricamento, una trasformazione base e il salvataggio.
    Include concetti di gestione memoria e flag di decodifica.
    """
    # 1. LETTURA (cv2.imread)
    # Teoricamente: imread decodifica i flussi di byte compressi (JPG/PNG) in array NumPy.
    # Flag IMREAD_COLOR: Carica a 8-bit per canale (0-255), scarta la trasparenza.
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # Controllo robustezza: imread non solleva eccezioni ma restituisce None se il path è errato
    if img is None:
        print(f"[-] Errore: Impossibile caricare {image_path.name}. File corrotto o percorso non valido.")
        return

    # 2. VISUALIZZAZIONE (Solo per debug, non consigliata in batch massivi)
    # Nota: Il nome della finestra 'Anteprima' serve come identificatore nel Window Manager
    cv2.imshow('Anteprima', img)
    
    # waitKey(500) blocca l'esecuzione per 500ms permettendo al sistema operativo di renderizzare la finestra.
    # Se premuto un tasto prima dei 500ms, restituisce il codice ASCII del tasto.
    cv2.waitKey(500)

    # 3. TRASFORMAZIONE (Esempio: Conversione in scala di grigi)
    # La formula della luminanza pesata è: Y = 0.299*R + 0.587*G + 0.114*B
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. SALVATAGGIO (cv2.imwrite)
    # Best Practice: Usiamo il formato PNG per evitare artefatti di compressione durante lo sviluppo
    # o JPG con parametri di qualità definiti per risparmiare spazio nel dataset finale.
    target_path = output_folder / f"processed_{image_path.stem}.jpg"
    
    # Specifichiamo la qualità JPG (da 0 a 100). Default è 95. 
    # IMWRITE_JPEG_QUALITY influenza il quantizzatore nella trasformata discreta del coseno (DCT).
    success = cv2.imwrite(str(target_path), gray_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    if success:
        print(f"[+] Salvato: {target_path.name}")

def batch_processor(source_dir: str, target_dir: str):
    """
    Scansiona una cartella e processa tutte le immagini in modo automatizzato.
    """
    # Pathlib permette una gestione dei percorsi cross-platform (Windows/Linux/Mac)
    src_path = Path(source_dir)
    dst_path = Path(target_dir)

    # Crea la cartella di destinazione se non esiste (Best practice: exist_ok=True)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Definiamo le estensioni supportate per evitare file di sistema (es. .DS_Store o .json)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Iterazione efficiente sui file della directory
    print(f"[*] Inizio processamento nella cartella: {src_path}")
    for file in src_path.iterdir():
        if file.suffix.lower() in valid_extensions:
            process_image_pipeline(file, dst_path)
    
    # Pulizia finale: chiude tutte le finestre di sistema aperte da OpenCV
    cv2.destroyAllWindows()
    print("[*] Batch processing completato.")

if __name__ == "__main__":
    
    # Definizione cartelle
    raw_folder = "dataset_raw"
    processed_folder = "dataset_processed"

    # 1. Scarichiamo le immagini (Nuova funzione)
    download_demo_images(raw_folder, count=2)

    # 2. Eseguiamo il processamento batch originale
    batch_processor(raw_folder, processed_folder)