import os

# --- [FASE 1]: CONFIGURAZIONE DEL MOTORE ---
# Spieghiamo al computer quale "motore" usare per far girare l'Intelligenza Artificiale.
# Usiamo TensorFlow perché è molto stabile per caricare modelli già pronti.
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras
import keras_cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO

class YOLONMSVisualizer:
    """
    Questa classe è come un "laboratorio" che usa YOLOv8 (un'IA molto famosa)
    per trovare oggetti nelle foto e decidere quale sia il riquadro migliore.
    """

    def __init__(self, model_preset="yolo_v8_m_pascalvoc"):
        """
        [PASSO 1.1]: Accendiamo il cervello dell'IA.
        Scarichiamo un modello pre-addestrato che sa già riconoscere oggetti comuni.
        """
        print(f"Caricamento del cervello IA ({model_preset})...")
        self.model = keras_cv.models.YOLOV8Detector.from_preset(
            model_preset,
            bounding_box_format="xyxy", # x_min, y_min, x_max, y_max
        )

    def calculate_iou(self, box1, box2):
        """
        [PASSO 2]: La matematica della "Sovrapposizione" (IoU).
        L'IoU (Intersection over Union) ci dice quanto due rettangoli sono sovrapposti.
        Serve per capire se due riquadri stanno guardando lo stesso identico oggetto.
        """
        # Troviamo i confini della zona dove i due rettangoli si toccano
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # Se i rettangoli non si toccano affatto
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        # Calcoliamo l'area della zona di scontro (intersezione)
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calcoliamo l'area totale dei due rettangoli
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Formula: Area di scontro diviso Area Totale (meno il doppione)
        union_area = area1 + area2 - intersection_area
        return intersection_area / union_area

    def load_image(self, url):
        """
        [PASSO 3]: Recupero della foto.
        Scarichiamo l'immagine da internet e la adattiamo alla misura 640x640,
        che è quella preferita dall'IA.
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        
        img = Image.open(BytesIO(res.content)).convert("RGB")
        return np.array(img.resize((640, 640)))

    def perform_analysis(self, image_url):
        """
        [FASE 2]: L'INFERENZA (L'IA entra in azione).
        Qui diamo la foto all'IA e lei ci restituisce migliaia di ipotesi su dove siano gli oggetti.
        """
        image = self.load_image(image_url)
        
        # Prepariamo la foto per l'IA (aggiungiamo una dimensione "batch")
        input_data = np.expand_dims(image, axis=0)
        
        print("L'IA sta analizzando la foto... attendere...")
        output = self.model.predict(input_data)
        
        # Prendiamo i risultati: riquadri (boxes) e punteggi di sicurezza (confidence)
        boxes = output['boxes'][0]
        confs = output['confidence'][0]
        
        # Filtriamo: teniamo solo i riquadri con almeno il 10% di sicurezza
        # per vedere la "nuvola" di tentativi che l'IA fa prima di decidere.
        mask = confs > 0.1
        candidates = boxes[mask]
        
        if len(boxes) > 0:
            # Il primo riquadro [0] è il "Vincitore" scelto dall'algoritmo NMS
            self.plot_results(image, candidates, boxes[0])
        else:
            print("Nessun oggetto trovato.")

    def plot_results(self, image, candidates, final_box):
        """
        [FASE 3]: VISUALIZZAZIONE GRAFICA.
        Creiamo un disegno che mostra il processo decisionale dell'IA.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # Usiamo una scala di colori (da blu a rosso) per l'IoU
        cmap = plt.colormaps.get_cmap('turbo')

        # DISEGNIAMO TUTTI I CANDIDATI (I "perdenti")
        for i, box in enumerate(candidates):
            score = self.calculate_iou(box, final_box)
            
            # Se è quasi identico al vincitore, non lo disegniamo come candidato
            if score > 0.99: continue

            color = cmap(score)
            # Disegniamo rettangoli sottili e trasparenti
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=0.5)
            ax.add_patch(rect)
            
            # Mostriamo il punteggio di sovrapposizione vicino al box
            if score > 0.05:
                ax.text(box[0], box[1] + (i % 5) * 10, f"{score:.2f}", color='white',
                        fontsize=7, bbox=dict(facecolor=color, alpha=0.7, pad=0))

        # DISEGNIAMO IL VINCITORE (Il riquadro finale scelto)
        f_rect = patches.Rectangle((final_box[0], final_box[1]), final_box[2]-final_box[0], 
                                 final_box[3]-final_box[1], linewidth=4, 
                                 edgecolor='#00FF00', facecolor='none', zorder=10)
        ax.add_patch(f_rect)
        
        # Mettiamo un'etichetta grande e chiara
        ax.text(final_box[0], final_box[1]-10, "VINCITORE (Miglior Match)", color='white',
                fontsize=10, fontweight='bold', backgroundcolor='#00FF00', zorder=11)

        plt.title("Processo di Selezione IA: I colori mostrano la sovrapposizione (IoU)")
        plt.axis('off')
        
        # Salviamo il risultato finale
        plt.savefig("risultato_yolo_iou.png", bbox_inches='tight')
        print("Analisi completata! Guarda il file 'risultato_yolo_iou.png'")
        plt.show()

# --- [AVVIO DEL PROGRAMMA] ---
if __name__ == "__main__":
    # Usiamo una famosa immagine di test (cane, bicicletta, auto)
    URL_TEST = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg"
    
    # Creiamo l'analizzatore e lanciamo l'IA sulla foto
    analizzatore = YOLONMSVisualizer()
    analizzatore.perform_analysis(URL_TEST)