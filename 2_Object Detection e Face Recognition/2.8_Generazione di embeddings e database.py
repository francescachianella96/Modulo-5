"""
SISTEMA DI RICONOSCIMENTO FACCIALE 
----------------------------------------------------------------------
Questo script implementa un pipeline riconoscimento biometrico.
Il sistema non si limita a confrontare immagini (pixel-by-pixel), ma estrae
una rappresentazione matematica astratta del volto.

COMPONENTI LOGICI:
1. INPUT: Caricamento immagini tramite 'face_recognition'.
2. EMBEDDING: Trasformazione del volto in un vettore numerico (128-d).
3. DATABASE: Archiviazione persistente dei vettori tramite 'pickle'.
4. INFERENZA: Confronto tra vettori tramite Distanza Euclidea.
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# ==========================================================
# 1. SEZIONE CONFIGURAZIONE
# ==========================================================
# BASE_DIR identifica dinamicamente la cartella dove si trova questo script.
# Questo garantisce che il programma trovi le immagini se salvate nella stessa cartella.
BASE_DIR = Path(__file__).resolve().parent

# PATH_DATABASE: Percorso del file binario dove verranno archiviati i volti noti.
PATH_DATABASE = BASE_DIR / "face_vault_2026.pkl"

# NOME_SOGGETTO: Etichetta testuale assegnata alla persona durante la registrazione.
NOME_SOGGETTO = "Elon Musk"

# PATH_REGISTRAZIONE: L'immagine sorgente usata come riferimento nel database.
PATH_REGISTRAZIONE = BASE_DIR / "download.jpg"

# PATH_TEST: L'immagine 'ignota' che il sistema deve provare a riconoscere.
PATH_TEST = BASE_DIR / "download (1).jpg"
# ==========================================================

# Configurazione Backend: Keras 3 è l'interfaccia standard moderna.
# Impostiamo PyTorch come motore di calcolo per ottimizzare le operazioni sui tensori.
os.environ["KERAS_BACKEND"] = "torch"
import keras
import face_recognition

class FaceIDSystem2026:
    """
    Agisce come 'Orchestratore' del sistema.
    Gestisce l'interazione tra i file su disco (Database .pkl) e 
    la memoria RAM (self.database), coordinando i modelli di Computer Vision.
    """
    
    def __init__(self, db_path: Path):
        """
        Costruttore del sistema. Inizializza il database e verifica il backend.
        
        Args:
            db_path (Path): Oggetto Path che punta al file del database .pkl
        """
        self.db_path = db_path
        # All'avvio, carichiamo subito i dati dal disco alla RAM (dizionario).
        # Questo permette di fare confronti istantanei senza leggere ogni volta il file.
        self.database: Dict[str, np.ndarray] = self._load_database()
        
        print(f"--- LOG: Sistema Inizializzato (Backend: {keras.backend.backend()}) ---")

    def _load_database(self) -> Dict:
        """
        Legge il file .pkl e ricostruisce l'oggetto Python originario.
        Interazione: RAM <--- DISCO (Deserializzazione)
        """
        if self.db_path.exists():
            with open(self.db_path, 'rb') as f:
                try:
                    # Carica il dizionario degli embeddings già salvati.
                    data = pickle.load(f)
                    return data if isinstance(data, dict) else {}
                except Exception as e:
                    print(f"Errore caricamento: {e}")
                    return {}
        return {}

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Carica un file immagine e lo converte in una matrice di pixel RGB.
        
        Args:
            path (Path): Percorso del file immagine.
            
        Returns:
            Optional[np.ndarray]: Array NumPy dei pixel o None se il file manca.
        """
        if not path.exists():
            print(f"Errore: Il file '{path.name}' non è stato trovato in {path.parent}")
            return None
        try:
            # Utilizza la utility di face_recognition per garantire il formato corretto.
            return face_recognition.load_image_file(str(path))
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine {path.name}: {e}")
            return None

    def _get_embedding(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """
        IL MOTORE AI: Estrae le 'Feature' (Caratteristiche).
        Questa riga invoca una ResNet pre-addestrata che:
        1. Trova il volto nell'immagine.
        2. Allinea il volto (ruota occhi/menton per averlo dritto).
        3. Genera 128 numeri che descrivono le distanze spaziali tra i tratti somatici.
        """
        # encodings sarà una lista di vettori (uno per ogni faccia trovata).
        encodings = face_recognition.face_encodings(image_array)
        
        # Restituiamo solo il primo volto trovato (indice 0).
        return encodings[0] if encodings else None

    def add_identity(self, name: str, image_path: Path):
        """
        Registra una nuova persona estraendo il suo embedding e salvandolo nel vault.
        """
        if name in self.database:
            print(f"Identità '{name}' già presente nel database.")
            return

        print(f"Registrazione in corso: {name}...")
        img_array = self._load_image(image_path)
        
        if img_array is not None:
            embedding = self._get_embedding(img_array)
            if embedding is not None:
                # Archiviazione dell'embedding (128 numeri) invece dell'intera immagine.
                self.database[name] = embedding
                self.save_db()
                print(f"OK: {name} registrato con successo nel vault.")
            else:
                print(f"Errore: Nessun volto rilevato in {image_path.name}.")

    def save_db(self):
        """
        Persiste il dizionario degli embeddings su disco in formato binario.
        """
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)

    def identify(self, image_path: Path, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Fase di Inferenza (Confronto):
        Confronta il volto ignoto con TUTTI i volti nel vault.
        
        Logica Matematica:
        Viene usata la 'Distanza Euclidea'. Se la distanza è 0, i volti sono identici.
        Più la distanza cresce, meno i due volti sono simili.
        """
        # 1. Carichiamo l'immagine di test
        test_img = self._load_image(image_path)
        if test_img is None: return "File Invalido", 0.0

        # 2. Generiamo l'embedding del volto da identificare
        unknown_encoding = self._get_embedding(test_img)
        if unknown_encoding is None: return "Nessun Volto Trovato", 0.0

        # 3. Prepariamo i dati del vault per il confronto massivo
        names = list(self.database.keys())
        known_encodings = np.array(list(self.database.values()))
        
        if not names: return "Database Vuoto", 0.0

        # 4. CALCOLO DELLE DISTANZE (Linear Algebra Optimization)
        # face_distance calcola la distanza tra il vettore ignoto e TUTTI quelli del DB contemporaneamente.
        distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        
        # 5. np.argmin trova l'indice del numero più piccolo nel vettore 'distances'.
        # Quel numero rappresenta il nostro 'Best Match'.
        best_match_idx = np.argmin(distances)
        
        # Trasformiamo la distanza (0.0-1.0) in una percentuale di confidenza (es. 0.2 dist -> 80% conf).
        confidence = 1 - distances[best_match_idx]

        # 6. Verifica della soglia (Tolerance)
        # Se anche la distanza più bassa è troppo alta, diciamo "Sconosciuto".
        if distances[best_match_idx] <= threshold:
            return names[best_match_idx], confidence
            
        return "Sconosciuto", confidence

def main():
    """
    Punto di ingresso dello script. Coordina le fasi di registrazione e test.
    """
    # STEP 0: Inizializzazione
    face_system = FaceIDSystem2026(PATH_DATABASE)

    # --- FASE 1: REGISTRAZIONE ---
    # Se la foto di registrazione esiste e il soggetto non è nel database, procediamo.
    if PATH_REGISTRAZIONE.exists():
        face_system.add_identity(NOME_SOGGETTO, PATH_REGISTRAZIONE)
    else:
        print(f"Nota: Foto registrazione '{PATH_REGISTRAZIONE.name}' non trovata. Controllo database esistente...")

    # --- FASE 2: RICONOSCIMENTO ---
    print("\n" + "="*45)
    print("      SESSIONE RICONOSCIMENTO LOCALE 2026")
    print("="*45)

    if PATH_TEST.exists():
        # Eseguiamo l'inferenza sull'immagine di test.
        nome, conf = face_system.identify(PATH_TEST)
        print(f"SOGGETTO RILEVATO: {nome}")
        print(f"LIVELLO CONFIDENZA: {conf:.2%}")
        
        # Messaggio di feedback basato sull'esito.
        if nome != "Sconosciuto" and nome != "Nessun Volto":
            print(f"Esito: Identificazione di '{nome}' completata con successo.")
        else:
            print("Esito: Soggetto non presente nel vault o volto non chiaro.")
    else:
        print(f"Errore: File di test '{PATH_TEST.name}' non trovato.")

    print("="*45)

if __name__ == "__main__":
    main()