import spacy
import os

def carica_modello_lg():
    """
    Carica il modello linguistico 'large' per l'italiano.
    Il modello 'lg' è più preciso grazie a vettori di parole più grandi.
    """
    try:
        return spacy.load("it_core_news_lg")
    except OSError:
        print("Modello 'it_core_news_lg' non trovato. Download in corso...")
        os.system("python -m spacy download it_core_news_lg")
        return spacy.load("it_core_news_lg")

# Inizializziamo il motore NLP
nlp = carica_modello_lg()

def identifica_root_e_soggetto(testo):
    """
    Identifica automaticamente la Radice (Root) e il Soggetto (nsubj).
    """
    doc = nlp(testo)
    root = None
    soggetto = None

    # Individuiamo la ROOT
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    
    # Cerchiamo il soggetto tra i figli della ROOT
    if root:
        for child in root.children:
            if child.dep_ == "nsubj":
                soggetto = child
                break
    
    return root, soggetto

def analizza_aggettivi_amod(testo):
    """
    Filtra la frase per mostrare solo i token con relazione 'amod'.
    """
    doc = nlp(testo)
    aggettivi = [token for token in doc if token.dep_ == "amod"]
    return aggettivi

if __name__ == "__main__":
    frase = "I ricercatori dell'università hanno pubblicato un nuovo studio fondamentale."
    
    print(f"FRASE DA ANALIZZARE: '{frase}'\n")

    # Task 1: Identificazione Root e Soggetto
    root, soggetto = identifica_root_e_soggetto(frase)
    print(f"[TASK 1] Analisi Strutturale:")
    print(f"-> Radice (Root): {root.text} (Pos: {root.pos_})")
    print(f"-> Soggetto (nsubj): {soggetto.text if soggetto else 'Non trovato'}")
    
    # Task 2: Filtro amod
    aggettivi_trovati = analizza_aggettivi_amod(frase)
    print(f"\n[TASK 2] Filtro Modificatori Aggettivali (amod):")
    if aggettivi_trovati:
        for adj in aggettivi_trovati:
            print(f"-> Token trovato: '{adj.text}' (collegato a: '{adj.head.text}')")
    else:
        print("-> Nessun modificatore aggettivale trovato.")

    # Conclusione didattica
    print("\n--- RIFLESSIONE SUGLI AGGETTIVI ---")
    print("La relazione 'amod' (adjectival modifier) ci permette di isolare le qualità specifiche degli oggetti.")
    print("In questa frase, 'nuovo' e 'fondamentale' modificano 'studio'.")
    print("Senza gli aggettivi, sapremmo che è stato pubblicato uno studio, ma perderemmo")
    print("l'informazione cruciale sulla sua importanza e novità, che è il valore aggiunto del testo.")