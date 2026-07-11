import os
import sys
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from typing import List, Tuple, Optional

# --- CONFIGURAZIONE BACKEND  ---
# Impostiamo PyTorch come motore di calcolo per Keras 3.
os.environ["KERAS_BACKEND"] = "torch"
import keras

# --- IMPORTAZIONE MEDIAPIPE TASKS ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceAnalysisEngine:
    """
    Motore biometrico avanzato per il monitoraggio della vigilanza.
    Implementa il calcolo di EAR (occhi) e MAR (bocca) tramite MediaPipe Tasks API.
    """
    
    def __init__(self, model_path: str = "face_landmarker.task"):
        """
        Inizializza il task di landmarking e gestisce il download del modello.
        """
        self.model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        self._ensure_model_exists(model_path)

        # Configurazione delle opzioni per il Face Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        try:
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("MediaPipe Face Landmarker (Tasks API) inizializzato con successo.")
        except Exception as e:
            print(f"Errore critico durante l'inizializzazione: {e}")
            self.detector = None

        # --- INDICI MESH A 478 PUNTI ---
        # Occhi (per EAR)
        self.EYE_LEFT = [362, 385, 387, 263, 373, 380]
        self.EYE_RIGHT = [33, 160, 158, 133, 153, 144]
        
        # Centri iride (per allineamento)
        self.IRIS_LEFT = 468
        self.IRIS_RIGHT = 473

        # --- SOLUZIONE TASK 1: INDICI BOCCA (MAR) ---
        self.MOUTH_TOP = 13       # Centro labbro superiore (interno)
        self.MOUTH_BOTTOM = 14    # Centro labbro inferiore (interno)
        self.MOUTH_LEFT = 78      # Angolo sinistro bocca
        self.MOUTH_RIGHT = 308    # Angolo destro bocca

    def _ensure_model_exists(self, path: str):
        """Scarica il file .task se non è presente nella cartella di lavoro."""
        if not os.path.exists(path):
            print(f"Modello '{path}' non trovato. Download in corso...")
            response = requests.get(self.model_url, stream=True)
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download completato.")

    def fetch_image(self, url: str) -> np.ndarray:
        """Scarica l'immagine e la converte in formato BGR per OpenCV."""
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def get_ear(self, landmarks, eye_indices: List[int], w: int, h: int) -> float:
        """Calcola l'Eye Aspect Ratio per monitorare la chiusura degli occhi."""
        p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
        v1 = np.linalg.norm(p[1] - p[5])
        v2 = np.linalg.norm(p[2] - p[4])
        horiz = np.linalg.norm(p[0] - p[3])
        return (v1 + v2) / (2.0 * horiz) if horiz > 0 else 0.0

    # --- SOLUZIONE TASK 2: METODO GET_MAR ---
    def get_mar(self, landmarks, w: int, h: int) -> float:
        """
        Calcola il Mouth Aspect Ratio (MAR) per rilevare lo sbadiglio.
        Formula: MAR = (distanza verticale labbra) / (distanza orizzontale angoli)
        """
        # Estrazione coordinate pixel dei punti chiave della bocca
        p_top = np.array([landmarks[self.MOUTH_TOP].x * w, landmarks[self.MOUTH_TOP].y * h])
        p_bottom = np.array([landmarks[self.MOUTH_BOTTOM].x * w, landmarks[self.MOUTH_BOTTOM].y * h])
        p_left = np.array([landmarks[self.MOUTH_LEFT].x * w, landmarks[self.MOUTH_LEFT].y * h])
        p_right = np.array([landmarks[self.MOUTH_RIGHT].x * w, landmarks[self.MOUTH_RIGHT].y * h])
        
        # Calcolo distanze euclidee
        vertical_dist = np.linalg.norm(p_top - p_bottom)
        horizontal_dist = np.linalg.norm(p_left - p_right)
        
        # Ritorna il rapporto (Ratio)
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.0

    def align_face(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Raddrizza il volto basandosi sull'asse degli occhi."""
        h, w = image.shape[:2]
        l_c = (landmarks[self.IRIS_LEFT].x * w, landmarks[self.IRIS_LEFT].y * h)
        r_c = (landmarks[self.IRIS_RIGHT].x * w, landmarks[self.IRIS_RIGHT].y * h)
        angle = np.degrees(np.arctan2(r_c[1] - l_c[1], r_c[0] - l_c[0]))
        center = (int((l_c[0] + r_c[0]) / 2), int((l_c[1] + r_c[1]) / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    def analyze(self, image_url: str):
        """Pipeline: Download -> Allineamento -> EAR -> MAR -> UI."""
        img_bgr = self.fetch_image(image_url)
        h, w = img_bgr.shape[:2]
        
        # Prima passata per allineamento
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        res = self.detector.detect(mp_image)
        if not res.face_landmarks: return

        # Allineamento e normalizzazione geometrica
        aligned_img = self.align_face(img_bgr, res.face_landmarks[0])
        
        # Seconda passata su immagine allineata per metriche biometriche stabili
        mp_aligned = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        res_aligned = self.detector.detect(mp_aligned)
        
        if res_aligned.face_landmarks:
            final_lms = res_aligned.face_landmarks[0]
            
            # --- SOLUZIONE TASK 3: CALCOLO E VISUALIZZAZIONE MAR ---
            ear_l = self.get_ear(final_lms, self.EYE_LEFT, w, h)
            ear_r = self.get_ear(final_lms, self.EYE_RIGHT, w, h)
            avg_ear = (ear_l + ear_r) / 2.0
            
            current_mar = self.get_mar(final_lms, w, h)
            
            # Rendering dei risultati grafici
            self._draw_ui(aligned_img, final_lms, avg_ear, current_mar)
            
            # Output finale
            Image.fromarray(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)).show()
            cv2.imwrite("biometric_analysis_result.jpg", aligned_img)
            print(f"Analisi conclusa. EAR: {avg_ear:.3f}, MAR: {current_mar:.3f}")

    def _draw_ui(self, img, lms, ear, mar):
        """Disegna i landmark e le info biometriche sullo schermo."""
        h, w = img.shape[:2]
        for p in lms:
            cv2.circle(img, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)
        
        # Info EAR (Occhi)
        e_color = (0, 255, 0) if ear > 0.22 else (0, 0, 255)
        cv2.putText(img, f"EAR (Occhi): {ear:.3f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, e_color, 2)
        
        # Info MAR (Bocca)
        m_color = (0, 255, 0) if mar < 0.5 else (0, 0, 255)
        cv2.putText(img, f"MAR (Bocca): {mar:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, m_color, 2)
        
        # Avviso Sbadiglio
        if mar > 0.5:
            cv2.putText(img, "YAWNING DETECTED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# --- ESECUZIONE ---
if __name__ == "__main__":
    engine = FaceAnalysisEngine()
    if engine.detector:
        # Immagine di test (puoi provare con URL di persone che sbadigliano)
        TEST_URL = "https://images.pexels.com/photos/2379004/pexels-photo-2379004.jpeg"
        engine.analyze(TEST_URL)