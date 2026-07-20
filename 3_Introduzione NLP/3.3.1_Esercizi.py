import os
import spacy
from transformers import pipeline

def solve_travel_exercise():
    """
    Risoluzione dell'esercizio NLP: estrazione entità geografiche e analisi del sentiment.
    """
    # Frase target fornita dalla traccia
    frase = "Mario Rossi ha visitato il Colosseo a Roma e ne è rimasto entusiasta!"
    
    print(f"--- ANALISI FEEDBACK UTENTE ---")
    print(f"Frase: '{frase}'\n")

    # --- PARTE 1: ESTRAZIONE ENTITÀ CON SPACY ---
    # Caricamento del modello italiano con logica di fallback (Best Practice)
    try:
        nlp = spacy.load("it_core_news_sm")
    except OSError:
        print("[INFO]: Modello spaCy it_core_news_sm non trovato. Installazione in corso...")
        os.system("python -m spacy download it_core_news_sm")
        nlp = spacy.load("it_core_news_sm")

    doc = nlp(frase)
    
    # Estrazione filtrata: Solo LOC (Luoghi) o GPE (Entità Geopolitiche)
    # Nota: Nel modello italiano 'Roma' è GPE, 'Colosseo' è solitamente LOC o FAC.
    luoghi_rilevati = [ent.text for ent in doc.ents if ent.label_ in ["LOC", "GPE"]]
    
    print(f"[spaCy] Entità geografiche/luoghi rilevati: {luoghi_rilevati}")

    # --- PARTE 2: SENTIMENT ANALYSIS CON TRANSFORMERS ---
    # Forziamo framework="pt" per evitare conflitti con Keras 3 (Fix implementato in nlp_pipeline_fixed.py)
    print("\n[Hugging Face] Inizializzazione pipeline sentiment...")
    
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="dbmdz/bert-base-italian-xxl-cased", 
        framework="pt"
    )
    
    risultato_sentiment = sentiment_analyzer(frase)
    
    # Estrazione dei valori per una stampa leggibile
    label = risultato_sentiment[0]['label']
    score = risultato_sentiment[0]['score']
    
    # Verifica se il sentiment è positivo (il modello usa etichette come 'POSITIVE' o 'LABEL_1' a seconda del training)
    sentimento_positivo = "POSITIVO" if label in ["POSITIVE", "LABEL_1"] else "NEGATIVO/NEUTRO"
    
    print(f"Sentiment Rilevato: {sentimento_positivo} (Confidenza: {score:.4f})")
    print("-" * 40)

if __name__ == "__main__":
    solve_travel_exercise()