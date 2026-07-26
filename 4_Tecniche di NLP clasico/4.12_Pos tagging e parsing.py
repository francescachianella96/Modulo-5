"""
DESCRIZIONE: Guida pratica al Part-of-Speech (POS) Tagging e al Dependency Parsing.
             Questo script utilizza spaCy, lo standard industriale per l'NLP rapido e accurato,
             per mostrare come trasformare un testo "piatto" in una struttura gerarchica logica.
OBIETTIVO: Spiegare come le macchine identificano le categorie grammaticali e le relazioni
           tra le parole (Soggetto, Verbo, Oggetto) per "capire" il senso di una frase.
"""

import spacy       # Libreria principale per l'NLP (la nostra "fabbrica" di analisi)
import os          # Usato solo per scaricare il modello linguistico se manca

# --- FASE 1: PREPARAZIONE DEL MOTORE (Slide 5: Velocità Industriale) ---

def carica_modello_linguistico():
    """
    Carica il cervello linguistico di spaCy per l'italiano.
    
    Interazione: Il modello contiene pesi statistici e regole grammaticali 
    pre-addestrate su milioni di frasi per riconoscere i pattern della lingua.
    """
    try:
        # 'it_core_news_sm' è un modello piccolo (sm), veloce e ottimo per imparare.
        # È il nostro "detective" che analizzerà il contesto delle parole.
        return spacy.load("it_core_news_sm")
    except OSError:
        # Se il modello non è installato sul computer, lo scarichiamo al volo.
        print("Modello non trovato. Download in corso...")
        os.system("python -m spacy download it_core_news_sm")
        return spacy.load("it_core_news_sm")

# Inizializziamo il nostro "motore" nlp globally per usarlo nelle funzioni
nlp = carica_modello_linguistico()

# --- FASE 2: ANALISI MORFOLOGICA (Slide 4: Identità delle Parole) ---

def mostra_identita_parole(testo):
    """
    Esegue il POS Tagging: assegna un'etichetta grammaticale a ogni parola.
    
    Argomenti:
        testo (str): La frase che vogliamo analizzare.
    """
    # 1. Il motore 'nlp' elabora il testo e crea un oggetto 'doc' (documento).
    # Il 'doc' non è solo testo, è un contenitore di oggetti 'token' (parole) ricchi di dati.
    doc = nlp(testo)
    
    print(f"\n[ANALISI POS] Identificando i mattoncini della frase: '{testo}'")
    
    # Cicliamo su ogni parola (token) del documento
    for token in doc:
        # token.text: La parola originale
        # token.pos_ : L'etichetta UPOS (Universal Part-of-Speech) come VERB, NOUN, ADJ.
        # token.lemma_: La forma base (es: "mangiato" -> "mangiare"). Utile per normalizzare i dati.
        print(f"-> Parola: '{token.text:12}' | Tag POS: {token.pos_:8} | Base (Lemma): {token.lemma_}")

# --- FASE 3: ANALISI SINTATTICA (Slide 7-8: Gerarchia e Relazioni) ---

def mostra_relazioni_sintattiche(testo):
    """
    Esegue il Dependency Parsing: trova i legami "genitore-figlio" tra le parole.
    
    Argomenti:
        testo (str): La frase da mappare.
    """
    doc = nlp(testo)
    
    print(f"\n[ANALISI PARSING] Costruendo l'albero delle dipendenze:")
    
    for token in doc:
        # token.dep_ : Il tipo di relazione (es: nsubj = soggetto, obj = oggetto).
        # token.head : Il "genitore" sintattico della parola (la 'Testa' a cui è legata).
        # Nelle relazioni sintattiche, ogni parola punta a chi la governa.
        print(f"-> '{token.text:12}' è un {token.dep_:10} di --> '{token.head.text}'")

# --- FASE 4: ESTRAZIONE CONOSCENZA (Slide 14: Il Setaccio d'Oro) ---

def estrai_triplette_chiave(testo):
    """
    Naviga l'albero sintattico per isolare il cuore del messaggio: CHI FA COSA.
    Questo filtra il rumore (articoli, preposizioni) e tiene solo i concetti puri.
    """
    doc = nlp(testo)
    print(f"\n[SETACCIO D'ORO] Estraendo triplette di conoscenza (Soggetto, Azione, Oggetto):")
    
    for token in doc:
        # Identifichiamo il Verbo principale (che solitamente è la ROOT, la radice dell'albero)
        if token.pos_ == "VERB" or token.dep_ == "ROOT":
            
            # Cerchiamo tra i "figli" (children) del verbo chi ha il ruolo di soggetto e oggetto
            soggetto = ""
            oggetto = ""
            
            for child in token.children:
                if "subj" in child.dep_:    # Cerchiamo 'nsubj' (nominal subject)
                    soggetto = child.text
                if "obj" in child.dep_:     # Cerchiamo 'obj' (direct object)
                    oggetto = child.text
            
            # Se abbiamo trovato sia chi compie l'azione che chi la riceve, stampiamo la tripletta
            if soggetto and oggetto:
                print(f"   >>> Conoscenza trovata: [{soggetto}] --({token.lemma_})--> [{oggetto}]")

# --- ESECUZIONE DEL PROGRAMMA ---

if __name__ == "__main__":
    # Esempio 1: Una frase semplice che mostra la differenza di ruoli
    frase_1 = "Il programmatore scrive il codice."
    mostra_identita_parole(frase_1)
    mostra_relazioni_sintattiche(frase_1)
    estrai_triplette_chiave(frase_1)
    
    # Esempio 2: Una frase più complessa per vedere il parsing in azione (Slide 9: lunghe distanze)
    frase_2 = "Gli studenti di Deep Learning analizzano con attenzione i modelli linguistici."
    mostra_identita_parole(frase_2)
    mostra_relazioni_sintattiche(frase_2)
    estrai_triplette_chiave(frase_2)

    print("\n--- CONCLUSIONE DIDATTICA ---")
    print("1. Il POS Tagging ha identificato i ruoli (Nomi, Verbi).")
    print("2. Il Parsing ha collegato le parole tra loro creando una gerarchia.")
    print("3. Il Setaccio ha isolato i concetti chiave ignorando il 'rumore' grammaticale.")