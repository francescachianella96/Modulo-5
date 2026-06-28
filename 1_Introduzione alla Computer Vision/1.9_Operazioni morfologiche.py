import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Best Practice 2026: Configurazione Keras 3 con backend PyTorch per interoperabilità
os.environ["KERAS_BACKEND"] = "torch"
import keras

def morphological_processing_lab(url=None):
    # 0. SCARICAMENTO AUTOMATICO (Novità)
    # Se non viene fornito un URL, usiamo un'immagine standard ideale per la morfologia
    # In questo caso, un'immagine con rumore perfetta per testare Opening e Closing.
    if url is None:
        url = "https://picsum.photos/500/500?grayscale"
    
    # keras.utils.get_file scarica il file e restituisce il percorso locale
    image_path = keras.utils.get_file("sample_morphology.png", url)

    # 1. CARICAMENTO E BINARIZZAZIONE
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Errore nel caricamento dell'immagine scaricata da: {url}")

    # Usiamo il Thresholding di Otsu per trovare automaticamente il punto di separazione ottimale.
    # Teoria: Otsu minimizza la varianza intra-classe dei pixel neri e bianchi.
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 2. DEFINIZIONE DELL'ELEMENTO STRUTTURANTE (KERNEL)
    # Il kernel definisce la "forma" con cui analizziamo l'immagine.
    # Un kernel 5x5 ellittico è più "morbido" di uno rettangolare.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 3. OPENING (Erosione -> Dilatazione)
    # Teoria: Rimuove piccoli artefatti bianchi esterni (rumore 'sale').
    # L'erosione elimina i punti piccoli, la dilatazione riporta gli oggetti grandi alla dimensione originale.
    opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. CLOSING (Dilatazione -> Erosione)
    # Teoria: Chiude piccoli buchi neri interni agli oggetti (rumore 'pepe').
    # La dilatazione fonde i bordi vicini, l'erosione previene l'espansione eccessiva dell'oggetto.
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. SCHELETRIZZAZIONE (Algoritmo Iterativo)
    # Teoria: Consiste nel sottrarre ripetutamente l'apertura dell'immagine erosa.
    # È l'essenza topologica: riduce l'oggetto a linee spesse 1 pixel.
    skeleton = np.zeros(binary_img.shape, np.uint8)
    temp_img = binary_img.copy()
    
    # Elemento strutturante a croce (standard per scheletrizzazione)
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Step: Erosione, Apertura, Sottrazione
        eroded = cv2.erode(temp_img, cross_kernel)
        temp_open = cv2.dilate(eroded, cross_kernel)
        subset = cv2.subtract(temp_img, temp_open)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp_img = eroded.copy()
        
        # Se non ci sono più pixel da erodere, abbiamo finito
        if cv2.countNonZero(temp_img) == 0:
            break

    # 6. INTEGRAZIONE KERAS 3 (Pre-processing per CNN)
    # Convertiamo lo scheletro in un tensore per usarlo come feature spaziale.
    # Normalizziamo a [0, 1] come richiesto dai modelli PyTorch.
    skeleton_tensor = keras.ops.convert_to_tensor(skeleton, dtype="float32") / 255.0

    # VISUALIZZAZIONE
    titles = ['Originale Binaria', 'Opening (No Noise)', 'Closing (No Holes)', 'Skeleton']
    images = [binary_img, opening, closing, skeleton]
    
    plt.figure(figsize=(18, 6))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

    return skeleton_tensor

morphological_processing_lab()