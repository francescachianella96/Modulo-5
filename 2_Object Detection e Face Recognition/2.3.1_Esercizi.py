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
    Wrapper professionale per la classificazione d'immagine.
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

    def load_local_image(self, file_path):
        """
        Carica un'immagine dal disco locale e la converte in formato OpenCV (BGR).
        """
        print(f"Caricamento immagine da: {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Errore: Il file '{file_path}' non esiste.")
            return None
            
        try:
            # Caricamento diretto con OpenCV
            img = cv2.imread(file_path)
            
            if img is None:
                raise ValueError("Il file non è un'immagine valida o formato non supportato.")
            return img
        except Exception as e:
            print(f"Errore durante il caricamento: {e}")
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

    # 2. Definiamo 3 percorsi a immagini locali
    local_images = [
        "01.jpg",
        "02.jpg",
        "03.jpg"
    ]

    print(f"Avvio analisi su {len(local_images)} immagini locali...")

    for img_path in local_images:
        try:
            # Recupero immagine locale
            raw_img = classifier.load_local_image(img_path)
            
            if raw_img is not None:
                # Pre-processing
                processed_data = classifier.preprocess(raw_img)

                # Classificazione
                predictions = classifier.classify(processed_data)

                print("\n" + "="*40)
                print(f"   RISULTATI CLASSIFICAZIONE: {os.path.basename(img_path)}")
                print("="*40)
                for i, (imagenet_id, label, prob) in enumerate(predictions):
                    print(f"{i+1}. {label.upper():<20} {prob*100:>6.2f}%")
                print("="*40)

                # Visualizzazione rapida
                display_img = raw_img.copy()
                # Riduciamo l'immagine per la visualizzazione se troppo grande
                if display_img.shape[0] > 800:
                    scale = 800 / display_img.shape[0]
                    width = int(display_img.shape[1] * scale)
                    height = int(display_img.shape[0] * scale)
                    display_img = cv2.resize(display_img, (width, height))
                
                label_text = f"{predictions[0][1]} ({predictions[0][2]*100:.1f}%)"
                cv2.putText(display_img, label_text, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                window_name = f"Analisi: {os.path.basename(img_path)}"
                cv2.imshow(window_name, display_img)
                print("Premi un tasto per passare alla prossima immagine...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Saltata immagine: {img_path}")
        
        except Exception as e:
            print(f"Errore critico durante l'analisi di {img_path}: {e}")
            
    print("\nAnalisi completata.")