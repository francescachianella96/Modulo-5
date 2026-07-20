import os
import re
import nltk
from nltk.corpus import stopwords
from typing import List, Set

# --- CONFIGURAZIONE AMBIENTE ---
# Impostiamo il backend PyTorch per Keras 3 per coerenza con gli standard industriali
os.environ["KERAS_BACKEND"] = "torch"

import keras

# Download delle risorse NLTK
nltk.download('stopwords', quiet=True)

def clean_review_pipeline(raw_text: str) -> str:
    """
    Pipeline per pulizia di recensioni tecnologiche.
    Rimuove tag, URL e rumore di dominio preservando la logica delle critiche.
    """
    
    # 1. Normalizzazione (Case Folding)
    # Portiamo tutto in minuscolo per uniformare il confronto con le stopwords
    text = raw_text.lower()
    
    # 2. Rimozione Tag HTML specifici (es. <span>) e generici
    # Teoria: I tag HTML sono artefatti di formattazione che non hanno valore semantico
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Rimozione URL
    # Gli indirizzi web sono rumore ad alta entropia nei task di classificazione testuale
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Rimozione punteggiatura e caratteri speciali
    # Manteniamo le lettere accentate italiane (àèìòù)
    text = re.sub(r'[^a-zàèìòù\s]', '', text)
    
    # 5. Gestione Stopwords Personalizzata
    # Carichiamo la lista standard italiana
    stop_words_set: Set[str] = set(stopwords.words('italian'))
    
    # AGGIUNTA STOPWORDS DI DOMINIO (Obbligatorie da traccia)
    domain_stops = ["telefono", "smartphone", "cellulare"]
    stop_words_set.update(domain_stops)
    
    # ECCEZIONE CRITICA: Preserviamo la parola "ma"
    # Teoria: In una recensione, "ma" introduce la 'congiunzione avversativa' 
    # che separa i pregi dai difetti. Rimuoverla distruggerebbe il senso della critica.
    if 'ma' in stop_words_set:
        stop_words_set.remove('ma')
        
    # Preserviamo anche 'non' per non invertire il senso dei predicati
    if 'non' in stop_words_set:
        stop_words_set.remove('non')
    
    # 6. Tokenizzazione e Filtraggio
    tokens = text.split()
    cleaned_tokens = [w for w in tokens if w in ["ma", "non"] or w not in stop_words_set]
    
    return " ".join(cleaned_tokens)

# --- ESECUZIONE TEST ---

if __name__ == "__main__":
    # Testo fornito dalla traccia
    testo_input = "Il <span>telefono</span> è incredibile! Ma la batteria dura poco. Scarica il manuale qui: https://fix-it.com/manual"

    print("--- PIPELINE DI PULIZIA SMARTPHONE ---")
    print(f"Testo Originale:\n{testo_input}\n")

    testo_pulito = clean_review_pipeline(testo_input)

    print(f"Testo Pulito:\n{testo_pulito}")
    
    # Verifica dei requisiti
    print("\n--- VERIFICA REQUISITI ---")
    print(f"Parola 'ma' presente: {'ma' in testo_pulito.split()}")
    print(f"Parola 'telefono' rimossa: {'telefono' not in testo_pulito.split()}")
    print(f"Tag <span> rimossi: {'span' not in testo_pulito}")
    print(f"URL rimosse: {'http' not in testo_pulito}")