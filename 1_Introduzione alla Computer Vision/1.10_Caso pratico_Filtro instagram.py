import cv2
import numpy as np
import requests
import os
# Aggiungiamo matplotlib per la visualizzazione
import matplotlib.pyplot as plt

# Questo ci permette di integrare filtri deterministici in pipeline di Deep Learning.
os.environ["KERAS_BACKEND"] = "torch"
import keras

def create_vintage_filter_and_visualize():
    # --- 1. CARICAMENTO ---
    # Scarica, converte e decodifica in un solo passaggio
    # Nota: l'immagine è in BGR
    url = "https://picsum.photos/800"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status() # Solleva un'eccezione per codici di errore HTTP
        img_bgr = cv2.imdecode(np.frombuffer(resp.content, np.uint8), 1)
        if img_bgr is None: raise ValueError("Impossibile decodificare l'immagine.")
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        return None

# --- 2. MANIPOLAZIONE CURVE TRAMITE LUT (Look-Up Table) ---
    
    # Inizializziamo una matrice di 256 elementi (corrispondenti ai valori 0-255).
    # Il tipo 'uint8' è obbligatorio perché i pixel di un'immagine standard sono a 8 bit.
    lut = np.zeros((1, 256), dtype=np.uint8)
    
    for i in range(256):
        # Applichiamo una trasformazione lineare: y = mx + q
        # i: valore originale (input)
        # 30: offset (q) -> "solleva" il nero puro a un grigio scuro, creando l'effetto sbiadito.
        # (225/255): pendenza (m) -> comprime l'intervallo originale [0, 255] in [30, 255].
        # In questo modo, la dinamica dell'immagine si riduce, tipico delle vecchie foto.
        
        valore_mappato = 30 + (i * (225 / 255))
        
        # np.clip assicura che se il calcolo supera 255 (o scende sotto 0),
        # il valore venga bloccato entro i limiti validi per un'immagine a 8 bit.
        lut[0, i] = np.clip(valore_mappato, 0, 255) 

    # cv2.LUT è estremamente efficiente. Per ogni pixel dell'immagine:
    # 1. Legge il valore (es. 100)
    # 2. Va all'indice 100 della nostra tabella 'lut'
    # 3. Sostituisce il 100 con il nuovo valore (es. 118)
    # Complessità: O(N) dove N è il numero di pixel, ma con calcoli ridotti al minimo.
    img_faded = cv2.LUT(img_bgr, lut)

    # --- 3. SOVRAPPOSIZIONE TEXTURE (GRANA PELLICOLA) ---
    # Genera rumore con media 0 (neutro) e deviazione 15 (intensità della grana).
    # Simula la disposizione casuale dei cristalli d'argento nella fotografia analogica.
    noise = np.random.normal(0, 15, img_bgr.shape).astype(np.uint8)
    
    # Blending tramite addWeighted (90% immagine, 10% rumore)
    img_textured = cv2.addWeighted(img_faded, 0.9, noise, 0.1, 0)

    # --- 4. VIGNETTATURA (MAPPING SPAZIALE) ---
    
    # 1. Recuperiamo le dimensioni dell'immagine (Altezza e Larghezza).
    rows, cols = img_bgr.shape[:2]

    # 2. Generiamo due vettori 1D seguendo una curva Gaussiana (a campana).
    # Il sigma (cols/2) definisce quanto la "luce" è diffusa: più è alto, più il centro è ampio.
    kernel_x = cv2.getGaussianKernel(cols, cols/2) # Vettore orizzontale
    kernel_y = cv2.getGaussianKernel(rows, rows/2) # Vettore verticale

    # 3. Creiamo la maschera 2D tramite prodotto esterno (Outer Product).
    # Moltiplicando i due vettori otteniamo una matrice dove il centro è il valore massimo.
    kernel = kernel_y * kernel_x.T

    # 4. Normalizzazione: dividiamo per il valore massimo per portare il range a [0.0, 1.0].
    # Il centro sarà 1.0 (colore originale), i bordi saranno vicini allo 0.0 (nero).
    mask = kernel / kernel.max()
    
    # 5. Applicazione della maschera tramite NumPy Broadcasting.
    # mask[:, :, np.newaxis] aggiunge una dimensione per "coprire" i 3 canali BGR.
    # Moltiplichiamo l'immagine per la maschera per scurire i bordi in modo radiale.
    img_final_bgr = (img_textured * mask[:, :, np.newaxis]).astype(np.uint8)

    # --- 5. ESPORTAZIONE (Opzionale) ---
    # cv2.imwrite("filtered_vintage.png", img_final_bgr)

    # OpenCV usa BGR, Matplotlib usa RGB. Dobbiamo convertire per visualizzare correttamente.
    img_original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_final_rgb = cv2.cvtColor(img_final_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))

    # Subplot 1: Immagine Originale
    plt.subplot(1, 2, 1)
    plt.title("Originale (da URL)")
    plt.imshow(img_original_rgb)
    plt.axis('off') # Nasconde gli assi

    # Subplot 2: Immagine Filtrata
    plt.subplot(1, 2, 2)
    plt.title("Filtro Vintage Applicato")
    plt.imshow(img_final_rgb)
    plt.axis('off')

    plt.tight_layout()
    print("Mostrando il confronto... Chiudi la finestra del grafico per terminare lo script.")
    plt.show()
    # ==========================================

    # --- 6. WRAPPING IN KERAS 3 ---
    # Convertiamo l'output in un tensore Keras (Backend PyTorch) 
    # Normalizzazione [0, 1] per compatibilità con i modelli.
    tensor_img = keras.ops.convert_to_tensor(img_final_bgr, dtype="float32") / 255.0
    
    return tensor_img

# Esecuzione
final_tensor = create_vintage_filter_and_visualize()

if final_tensor is not None:
    print(f"\nProcesso completato. Shape del tensore Keras in uscita: {final_tensor.shape}")