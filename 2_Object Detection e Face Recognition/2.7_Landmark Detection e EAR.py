import os
import sys
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Optional

# --- CONFIGURAZIONE BACKEND ---
# Keras 3 è un framework agnostico. Impostiamo "torch" (PyTorch) come motore di calcolo
# per sfruttare l'accelerazione hardware e l'interoperabilità con i modelli moderni.
os.environ["KERAS_BACKEND"] = "torch"
import keras

# --- NUOVA IMPORTAZIONE MEDIAPIPE TASKS ---
# MediaPipe si è evoluto verso le Tasks API, che utilizzano bundle di modelli (.task)
# per un'inferenza più efficiente e cross-platform.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceAnalysisEngine:
    """
    Advanced biometric engine based on the MediaPipe Tasks API.

    This class handles the end-to-end pipeline for facial analysis, including:
    - Automatic management of the .task model bundle (download and initialization).
    - 3D Face Landmarking with 478 points (Face Mesh).
    - Geometric normalization (Pose Alignment) via affine transformations.
    - Vigilance analysis using the Eye Aspect Ratio (EAR).
    """
    
    def __init__(self, model_path: str = "face_landmarker.task"):
        """
        Initializes the biometric detector.

        Args:
            model_path (str): Local path to the 'face_landmarker.task' bundle.
                             If missing, it will be downloaded automatically.
        
        The initialization configures the vision task options, sets confidence
        thresholds for detection and tracking, and prepares the 478-point mesh indices.
        """
        # URL ufficiale per scaricare il bundle del modello se mancante
        self.model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        
        # Verifica se il modello esiste localmente, altrimenti lo scarica
        self._ensure_model_exists(model_path)

        # 1. BaseOptions: Configura le impostazioni di base come il percorso del modello AI (.task).
        # Il file .task contiene il modello TFLite e i metadati di pre/post-elaborazione.
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # 2. FaceLandmarkerOptions: Configura il comportamento specifico del rilevatore.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True, # Abilita il calcolo delle espressioni facciali (es. occhi chiusi)
            num_faces=1,                  # Limita l'analisi a un solo volto per ottimizzare le prestazioni
            min_face_detection_confidence=0.5, # Soglia minima per considerare rilevato un volto
            min_face_presence_confidence=0.5,  # Soglia minima per confermare la presenza del volto
            min_tracking_confidence=0.5        # Soglia minima per mantenere il tracciamento tra i frame
        )
        
        # Creazione del detector tramite le opzioni definite
        try:
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe Face Landmarker (Tasks API) inizializzato con successo.")
        except Exception as e:
            print(f"Errore critico durante l'inizializzazione: {e}")
            self.detector = None

        # --- INDICI MESH A 478 PUNTI ---
        # Indici delle palpebre per il calcolo dell'EAR (Eye Aspect Ratio)
        self.EYE_LEFT = [362, 385, 387, 263, 373, 380]
        self.EYE_RIGHT = [33, 160, 158, 133, 153, 144]
        
        # Indici dei centri delle iridi, utilizzati per l'allineamento della testa
        self.IRIS_LEFT = 468
        self.IRIS_RIGHT = 473

    def _ensure_model_exists(self, path: str):
        """
        Garantisce la presenza del bundle del modello richiesto nel filesystem locale.

        Se il file .task non viene trovato, avvia un download in streaming dai server ufficiali
        Google Cloud Storage. L'uso dello streaming garantisce l'efficienza della memoria
        anche per modelli di grandi dimensioni.

        Argomenti:
            path (str): Il percorso di destinazione previsto per il modello.
        """
        if not os.path.exists(path):
            print(f"Modello '{path}' non trovato. Download in corso dai server Google...")
            try:
                # Download a chunk per gestire file di grandi dimensioni senza saturare la RAM
                response = requests.get(self.model_url, stream=True, timeout=30)
                response.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download completato correttamente.")
            except Exception as e:
                print(f"Errore durante il download del modello: {e}")
                sys.exit(1)

    def fetch_image(self, url: str) -> np.ndarray:
        """
        Scarica un'immagine remota e la prepara per l'elaborazione con OpenCV.

        Argomenti:
            url (str): L'URL dell'immagine da recuperare.

        Ritorna:
            np.ndarray: L'immagine in formato BGR (Standard OpenCV).
        
        Nota: L'immagine viene convertita dal formato RGB di PIL a NumPy/BGR poiché
        OpenCV utilizza l'ordine dei canali BGR.
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        # OpenCV lavora in BGR, quindi invertiamo i canali
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def get_ear(self, landmarks, eye_indices: List[int], w: int, h: int) -> float:
        """
        Calcola l'Eye Aspect Ratio (EAR) per un singolo occhio.

        L'EAR è un valore scalare che correla con il livello di apertura dell'occhio.
        Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        dove:
            - p1, p4: angoli orizzontali dell'occhio.
            - p2, p3, p5, p6: punti verticali della palpebra.

        Argomenti:
            landmarks: Lista dei landmark facciali normalizzati da MediaPipe.
            eye_indices (List[int]): Indici specifici della mesh per l'occhio.
            w (int): Larghezza dell'immagine.
            h (int): Altezza dell'immagine.

        Ritorna:
            float: Il valore EAR calcolato. Un valore basso indica un occhio chiuso.
        """
        # Convertiamo i landmark normalizzati (0-1) in coordinate pixel reali
        p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
        
        # Distanze tra i punti superiori e inferiori della palpebra
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        
        # Distanza tra gli angoli esterni dell'occhio
        horiz = np.linalg.norm(p[0] - p[3])
        
        # Calcolo finale del rapporto
        return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0.0

    def align_face(self, image: np.ndarray, landmarks) -> np.ndarray:
        """
        Esegue la normalizzazione della posa (Allineamento del volto).

        Questo metodo calcola l'angolo di inclinazione basato sui centri delle iridi
        e applica una rotazione affine per rendere l'asse orizzontale del volto 
        parallelo all'asse orizzontale del frame.

        Questo passaggio è CRUCIALE per la precisione dell'EAR in quanto elimina
        la distorsione verticale causata dall'inclinazione della testa.

        Argomenti:
            image (np.ndarray): Il frame BGR originale.
            landmarks: Landmark grezzi dal primo rilevamento.

        Ritorna:
            np.ndarray: L'immagine ruotata (allineata).
        """
        h, w = image.shape[:2]
        l_iris = landmarks[self.IRIS_LEFT]
        r_iris = landmarks[self.IRIS_RIGHT]
        
        # Posizione dei centri delle iridi in pixel
        l_center = (l_iris.x * w, l_iris.y * h)
        r_center = (r_iris.x * w, r_iris.y * h)
        
        # Calcolo dell'angolo di inclinazione tramite arcotangente
        angle = np.degrees(np.arctan2(r_center[1] - l_center[1], r_center[0] - l_center[0]))
        
        # Centro della rotazione (punto medio tra gli occhi)
        eye_mid = (int((l_center[0] + r_center[0]) / 2), int((l_center[1] + r_center[1]) / 2))
        
        # Creazione della matrice di rotazione e applicazione del warping
        M = cv2.getRotationMatrix2D(eye_mid, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    def analyze(self, image_url: str):
        """
        Esegue l'intera pipeline di analisi biometrica.

        Flusso di lavoro:
        1. Fetch: Scarica l'immagine dall'URL fornito.
        2. Rilevamento Iniziale: Trova il volto e i centri degli occhi nell'immagine grezza.
        3. Allineamento: Ruota l'immagine per orizzontalizzare lo sguardo.
        4. Analisi di Precisione: Esegue nuovamente il landmarker sull'immagine ALLINEATA per la massima precisione EAR.
        5. Rendering UI: Disegna i landmark e visualizza lo stato (Sveglio/Affaticato).

        Argomenti:
            image_url (str): URL di origine della foto.
        """
        # 1. Recupero dati
        img_bgr = self.fetch_image(image_url)
        h, w = img_bgr.shape[:2]
        
        # Conversione obbligatoria per MediaPipe Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        
        # 2. Rilevamento iniziale
        detection_result = self.detector.detect(mp_image)
        if not detection_result.face_landmarks:
            print("Nessun volto rilevato nell'immagine.")
            return

        # Estraiamo i landmark del primo volto rilevato
        raw_lms = detection_result.face_landmarks[0]
        
        # 3. Normalizzazione geometrica del volto
        aligned_img = self.align_face(img_bgr, raw_lms)
        
        # 4. Analisi di precisione sull'immagine allineata
        mp_image_aligned = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        res_aligned = self.detector.detect(mp_image_aligned)
        
        if res_aligned.face_landmarks:
            final_lms = res_aligned.face_landmarks[0]
            
            # Calcolo dell'EAR per entrambi gli occhi e media
            ear_l = self.get_ear(final_lms, self.EYE_LEFT, w, h)
            ear_r = self.get_ear(final_lms, self.EYE_RIGHT, w, h)
            avg_ear = (ear_l + ear_r) / 2.0
            
            # 5. Rendering della User Interface grafica
            self._draw_ui(aligned_img, final_lms, avg_ear)
            
            # Mostriamo l'anteprima e salviamo il file
            Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)).show()
            cv2.imwrite("aligned_face_2026.jpg", aligned_img)
            print(f"Analisi completata con successo. EAR calcolato: {avg_ear:.4f}")

    def _draw_ui(self, img, lms, ear):
        """
        Disegna gli elementi grafici sul frame.
        Include i punti della mesh e le etichette di testo dinamiche.
        """
        h, w = img.shape[:2]
        
        # Disegno di ogni punto della Face Mesh (punti verdi)
        for p in lms:
            cv2.circle(img, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)
        
        # Logica di stato basata sulla soglia EAR (0.22 è un valore tipico per la sonnolenza)
        status = "ALERT (SVEGLIO)" if ear > 0.22 else "DROWSY (AFFATICATO)"
        color = (0, 255, 0) if ear > 0.22 else (0, 0, 255) # Verde se sveglio, Rosso se stanco
        
        # Stampa del testo informativo sull'immagine
        cv2.putText(img, f"EAR: {ear:.3f} | {status}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# --- BLOCCO DI ESECUZIONE ---
if __name__ == "__main__":
    # Inizializziamo il motore (scatenerà il download se necessario)
    engine = FaceAnalysisEngine()
    
    # Se il motore è pronto, eseguiamo l'analisi su una foto di test
    if engine.detector:
        # Immagine di test: volto maschile in alta risoluzione
        TEST_URL = "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg"
        engine.analyze(TEST_URL)