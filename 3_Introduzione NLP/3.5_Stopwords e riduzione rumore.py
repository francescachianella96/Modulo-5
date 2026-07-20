import os

# Configurazione del Backend per Keras 3 (Best Practice 2026)
# Impostiamo PyTorch come motore computazionale prima di caricare Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
import re
import nltk
from nltk.corpus import stopwords
from typing import List, Set

# Download delle risorse necessarie (eseguito solo la prima volta)
# NLTK rimane lo standard accademico per le liste di stopwords multilingua
nltk.download('stopwords')

def clean_text_pipeline(raw_text: str, custom_stops: List[str] = []) -> str:
    """
    Pipeline completa di pulizia e rimozione del rumore.
    
    Teoria: Il 'Rumore' è ogni informazione che non contribuisce alla 
    distribuzione semantica del testo. Rimuoverlo riduce la varianza dei dati.
    """
    
    # 1. Normalizzazione (Case Folding)
    # Fondamentale per evitare che 'Python' e 'python' siano visti come token diversi
    text = raw_text.lower()
    
    # 2. Rimozione Tag HTML
    # Teoria: I tag (es. <div>) sono rumore strutturale, non linguistico.
    # Usiamo una Regex che identifica tutto ciò che è racchiuso tra < e >
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Rimozione URL
    # Gli URL hanno entropia altissima ma valore semantico nullo in compiti generici.
    # Pattern per catturare http, https e www
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Rimozione caratteri speciali e punteggiatura
    # Manteniamo solo lettere e spazi. Nota: in certi casi (es. Sentiment Analysis)
    # i punti esclamativi potrebbero essere mantenuti. Qui puliamo tutto.
    text = re.sub(r'[^a-zàèìòù\s]', '', text)
    
    # 5. Gestione Stopwords
    # Carichiamo le stopwords italiane standard
    stop_words_set: Set[str] = set(stopwords.words('italian'))
    
    # Personalizzazione: Aggiunta di termini specifici del dominio (Customizzazione)
    # Teoria: Le liste standard sono generaliste; ogni dataset ha il suo rumore specifico.
    if custom_stops:
        stop_words_set.update(custom_stops)
    
    # Eccezione: Rimuoviamo 'non' dalle stopwords se vogliamo preservare la negazione
    # Teoria: In NLP 'non' è spesso una stopword, ma è vitale per il senso logico.
    if 'non' in stop_words_set:
        stop_words_set.remove('non')
    
    # 6. Tokenizzazione e Filtraggio
    # Dividiamo per spazi e rimuoviamo i termini se presenti nel set (Lookup O(1))
    tokens = text.split()
    cleaned_tokens = [w for w in tokens if w not in stop_words_set]
    
    # Ricostruiamo la stringa pulita
    return " ".join(cleaned_tokens)

# --- ESEMPIO DI UTILIZZO ---

raw_data = [
    "Il corso di AI è fantastico! <br> Visita https://ai-deeplearning.it per info.",
    "Non mi è piaciuto il modulo, troppo complesso e pieno di bug.",
    "L'intelligenza artificiale (AI) cambierà il mondo! 🤖 #AI2026"
]

# Definiamo stopwords specifiche per il nostro dominio (es. 'corso', 'modulo')
my_custom_stops = ['corso', 'modulo', 'info', 'ai']

print("--- Inizio Processamento Testi ---")
for doc in raw_data:
    clean_doc = clean_text_pipeline(doc, custom_stops=my_custom_stops)
    print(f"Originale: {doc}")
    print(f"Pulito:    {clean_doc}\n")
