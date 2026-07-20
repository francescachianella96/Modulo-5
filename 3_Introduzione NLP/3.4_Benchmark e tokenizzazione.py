import os  # Importiamo il modulo per gestire i comandi di sistema (es. download modelli)
import time # Importiamo time per calcolare la latenza dei diversi approcci
import nltk # Libreria storica per il processamento del linguaggio naturale
import spacy # Framework industriale moderno ottimizzato per le performance
from spacy.symbols import ORTH # Importiamo il simbolo ORTH per definire regole ortografiche custom

def run_nlp_benchmark():
    # Definiamo un testo complesso con abbreviazioni legali e punteggiatura ambigua
    # La sfida è non dividere abbreviazioni come "c.p.c." o "D.Lgs."
    text = "L'art. 24 del c.p.c. definisce i termini per l'appello nel D.Lgs. 231/01."

    print(f"--- ANALISI DEL TESTO: {text} ---\n")

    # --- 1. APPROCCIO NLTK (Granulare/Regolistico) ---
    # Scarichiamo la risorsa 'punkt_tab', necessaria nelle versioni 2026 per la tokenizzazione
    nltk.download('punkt_tab', quiet=True) 
    
    # Registriamo il tempo di inizio per il benchmark NLTK
    start_time = time.time()
    
    # Tokenizzazione NLTK: usa l'algoritmo 'Punkt' per decidere dove spezzare le parole
    nltk_tokens = nltk.word_tokenize(text, language='italian')
    
    # Calcoliamo la durata dell'operazione
    nltk_duration = time.time() - start_time
    
    # Stampiamo i risultati e il tempo impiegato
    print(f"[NLTK] Token trovati ({len(nltk_tokens)}): {nltk_tokens}")
    print(f"[NLTK] Tempo di esecuzione: {nltk_duration:.6f}s\n")

    # --- 2. APPROCCIO SPACY (Industriale/Statistico) ---
    try:
        # Carichiamo il modello italiano 'small' pre-addestrato
        nlp = spacy.load("it_core_news_sm")
    except OSError:
        # Se il modello non è installato, eseguiamo il download via terminale
        os.system("python -m spacy download it_core_news_sm")
        # Ricarichiamo il modello dopo l'installazione
        nlp = spacy.load("it_core_news_sm")

    # Registriamo il tempo di inizio per il benchmark spaCy
    start_time = time.time()
    
    # Creiamo l'oggetto 'Doc': spaCy analizza il testo in un unico passaggio (pipeline)
    doc = nlp(text)
    
    # Estraiamo il testo di ogni token dall'oggetto Doc
    spacy_tokens = [token.text for token in doc]
    
    # Calcoliamo la durata dell'operazione
    spacy_duration = time.time() - start_time
    
    # Stampiamo i risultati: noterai che spaCy gestisce meglio gli apostrofi rispetto a NLTK
    print(f"[spaCy] Token trovati ({len(spacy_tokens)}): {spacy_tokens}")
    print(f"[spaCy] Tempo di esecuzione: {spacy_duration:.6f}s\n")

    # --- 3. PERSONALIZZAZIONE DEL TOKENIZER (Dominio Legale) ---
    # Definiamo la regola: quando trovi "c.p.c.", non dividerlo in "c.", "p.", "c."
    # ORTH definisce la stringa esatta che deve essere restituita come unico token
    special_case = [{ORTH: "c.p.c."}]
    
    # Aggiungiamo il caso speciale alla logica del tokenizer di spaCy
    nlp.tokenizer.add_special_case("c.p.c.", special_case)
    
    # Rieseguiamo l'analisi sullo stesso testo
    doc_custom = nlp(text)
    
    # Estraiamo nuovamente i token per verificare la modifica
    custom_tokens = [t.text for t in doc_custom]
    
    # Mostriamo come il numero di token sia diminuito poiché l'abbreviazione è ora unita
    print(f"[spaCy Custom] Dopo la regola 'c.p.c.': {custom_tokens}")
    print(f"[spaCy Custom] Nuova lunghezza: {len(custom_tokens)}")
    print("Nota: 'c.p.c.' è ora trattato come un'unica entità semantica.\n")

    # --- 4. ANALISI DEI METADATI (Deep Insight) ---
    # Mostriamo informazioni utili per il debugging dei modelli NLP
    print("[Insight] Analisi della struttura dei primi 5 token:")
    for token in list(doc_custom)[:5]:
        # Stampa: Testo | Is_Punct (è punteggiatura?) | Is_Abbrev (è abbreviazione?)
        print(f"Token: {token.text:12} | Punct: {str(token.is_punct):5} | Like_Num: {token.like_num}")

# Avviamo il benchmark se lo script viene eseguito direttamente
if __name__ == "__main__":
    run_nlp_benchmark()