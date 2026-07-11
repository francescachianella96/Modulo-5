import os
import requests
import numpy as np
import cv2
from io import BytesIO

# --- CONFIGURAZIONE BACKEND 2026 ---
# Impostiamo PyTorch come motore di calcolo prima dell'import di Keras.
os.environ["KERAS_BACKEND"] = "torch"

import keras

class ImageClassifierSOTA:
    """
    Wrapper per la classificazione d'immagine.
    Integra MobileNetV2 con pre-processing dinamico da URL web.
    """
    def __init__(self, model_name="MobileNetV2"):
        print(f"Inizializzazione modello: {model_name} con pesi ImageNet...")
        
        # Carichiamo MobileNetV2: architettura ottimizzata per l'efficienza.
        # Usa le 'Depthwise Separable Convolutions' per ridurre i parametri senza perdere troppa accuratezza.
        self.model = keras.applications.MobileNetV2(
            weights="imagenet", 
            include_top=True
        )
        
        # Dimensione standard per MobileNetV2: 224x224 pixel.
        self.input_shape = (224, 224)

    def download_image(self, url):
        """
        Scarica un'immagine da un URL e la converte in formato OpenCV (BGR).
        """
        print(f"Scaricamento immagine da: {url}...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Conversione dei byte in array numpy per OpenCV
            image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Il file scaricato non è un'immagine valida.")
            return img
        except Exception as e:
            print(f"Errore durante il download: {e}")
            return None

    def preprocess(self, img_bgr):
        """
        Trasforma l'immagine grezza in un tensore pronto per il Deep Learning.
        """
        # 1. Conversione BGR -> RGB (Cruciale per i modelli addestrati su ImageNet)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Resize: Adattamento alla risoluzione 224x224
        img_resized = cv2.resize(img_rgb, self.input_shape)
        
        # 3. Espansione Batch: Da (224, 224, 3) a (1, 224, 224, 3)
        img_batch = np.expand_dims(img_resized, axis=0).astype("float32")
        
        # 4. Normalizzazione: MobileNetV2 richiede pixel nel range [-1, 1]
        return keras.applications.mobilenet_v2.preprocess_input(img_batch)

    def classify(self, preprocessed_img, top_k=3):
        """
        Esegue l'inferenza e decodifica i risultati in classi leggibili.
        """
        # Inferenza tramite il backend selezionato (PyTorch)
        preds = self.model(preprocessed_img, training=False)
        
        # Spostiamo il risultato su CPU e convertiamo in Numpy per la decodifica
        preds_numpy = keras.ops.convert_to_numpy(preds)
        
        # Decode: Converte i vettori di probabilità in (ID, Etichetta, Probabilità)
        return keras.applications.mobilenet_v2.decode_predictions(preds_numpy, top=top_k)[0]

# --- WORKFLOW DI ESECUZIONE ---
if __name__ == "__main__":
    # 1. Istanza del classificatore
    classifier = ImageClassifierSOTA()

    # 2. URL di test (Esempio: un Golden Retriever da Unsplash)
    test_url = "https://images.unsplash.com/photo-1552053831-71594a27632d?q=80&w=800"

    try:
        # Recupero immagine
        raw_img = classifier.download_image(test_url)
        
        if raw_img is not None:
            # Pre-processing
            processed_data = classifier.preprocess(raw_img)

            # Classificazione
            predictions = classifier.classify(processed_data)

            print("\n" + "="*40)
            print("   RISULTATI CLASSIFICAZIONE SOTA")
            print("="*40)
            for i, (imagenet_id, label, prob) in enumerate(predictions):
                print(f"{i+1}. {label.upper():<20} {prob*100:>6.2f}%")
            print("="*40)

            # Visualizzazione rapida (facoltativa)
            cv2.putText(raw_img, f"Pred: {predictions[0][1]}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Classificazione Web", raw_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Errore critico nella pipeline: {e}")