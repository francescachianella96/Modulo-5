import os
import cv2
import numpy as np
from pathlib import Path
import requests

# Configurazione backend 
os.environ["KERAS_BACKEND"] = "torch"

def download_demo_images(target_dir: str, count: int = 2):
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"[*] Download di {count} immagini di test (formato .webp) in corso...")
    for i in range(count):
        # Nota: Picsum restituisce solitamente JPG, forziamo l'estensione webp per il test
        url = f"https://picsum.photos/640/480?random={i}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                file_path = path / f"test_image_{i}.webp" # Salviamo come .webp per la pipeline
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"[+] Scaricata: {file_path.name}")
        except Exception as e:
            print(f"[-] Errore: {e}")

def process_image_pipeline(image_path: Path, output_folder: Path) -> bool:
    """
    Ritorna True se l'immagine è stata processata, False se scartata.
    """
    # 1. LETTURA: IMREAD_UNCHANGED carica anche il canale Alpha (trasparenza)
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return False

    # 2. CONTROLLO DIMENSIONI: img.shape è (altezza, larghezza, canali)
    # Scartiamo se la larghezza (indice 1) è <= 500
    if img.shape[1] <= 500:
        return False

    # 3. CONVERSIONE IN BGRA:
    # Se l'immagine ha 3 canali (BGR), aggiungiamo il canale Alpha. 
    # Se ne ha già 4, è già BGRA.
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # 4. SALVATAGGIO: Formato .png (lossless di default in OpenCV)
    target_path = output_folder / f"{image_path.stem}.png"
    success = cv2.imwrite(str(target_path), img)
    
    if success:
        print(f"[+] Processata e salvata: {target_path.name}")
    return True

def batch_processor(source_dir: str, target_dir: str):
    src_path = Path(source_dir)
    dst_path = Path(target_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # REQUISITO: Solo file .webp
    valid_extensions = ('.webp',)
    
    discarded_count = 0
    processed_count = 0

    print(f"[*] Inizio Data Cleaning in: {src_path}")
    
    for file in src_path.iterdir():
        if file.suffix.lower() in valid_extensions:
            # Se la pipeline non processa l'immagine, la contiamo come scartata
            if process_image_pipeline(file, dst_path):
                processed_count += 1
            else:
                discarded_count += 1
    
    print("---")
    print(f"[*] Elaborazione completata.")
    print(f"[*] Immagini salvate: {processed_count}")
    print(f"[*] Immagini scartate (troppo piccole o errori): {discarded_count}")

if __name__ == "__main__":
    raw_folder = "dataset_raw"
    processed_folder = "dataset_processed"

    download_demo_images(raw_folder, count=3)
    batch_processor(raw_folder, processed_folder)