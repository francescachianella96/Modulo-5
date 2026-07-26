"""
================================================================================
FONDAMENTI DI NLP: NAMED ENTITY RECOGNITION (NER) CON SPACY
================================================================================
DESCRIZIONE ARCHITETTURALE:

1. INIZIALIZZAZIONE: Carichiamo il "modello linguistico", ovvero il cervello statistico (Classe Language).
2. ELABORAZIONE: Trasformiamo il testo grezzo (stringa) in un oggetto intelligente (Classe Doc).
3. ESTRAZIONE: Recuperiamo le entità nominate (nomi, luoghi, aziende) (Classe Span/Entity).
4. DECOSTRUZIONE: Analizziamo come il computer "vede" le entità tramite il sistema BIO (Classe Token).
5. VISUALIZZAZIONE: Creiamo un report grafico in HTML per rendere i dati leggibili agli umani.

INTERAZIONE TRA LE COMPONENTI:
- 'nlp' (Language): È il motore. Prende il testo e lo passa attraverso vari componenti (tokenizer, tagger, ner).
- 'doc' (Doc): È il risultato dell'elaborazione. Contiene il testo originale + tutti i metadati estratti.
- 'ent' (Span): Sono le "fette" del doc identificate come entità (es. "Elon Musk").
- 'token' (Token): Sono gli atomi del testo. Ogni singola parola o segno di punteggiatura.
================================================================================
"""

# Importiamo le librerie necessarie
import spacy                # La libreria principale per il Natural Language Processing
from spacy import displacy  # Sottogruppo di spacy dedicato alla visualizzazione grafica
import sys                  # Per gestire l'uscita forzata in caso di errore
import webbrowser           # Per aprire automaticamente il browser con i risultati
import os                   # Per gestire i percorsi dei file sul sistema

def carica_modello_nlp(nome_modello: str = "it_core_news_lg"):
    """
    Questa funzione prepara il 'motore' di intelligenza linguistica.
    """
    try:
        # Proviamo a caricare il modello specificato (di default quello 'large' per l'italiano)
        # nlp diventa un oggetto di classe 'Language' che contiene vocabolario e pesi neurali
        nlp = spacy.load(nome_modello)
        print(f"[SISTEMA] Modello '{nome_modello}' caricato e pronto all'uso.")
        return nlp
    except OSError:
        # Se il modello non è installato, diamo le istruzioni per scaricarlo
        print(f"[ERRORE] Il modello '{nome_modello}' non è stato trovato.")
        print("Esegui: 'python -m spacy download it_core_news_lg' nel tuo terminale.")
        sys.exit(1) # Esce dal programma perché non possiamo proseguire senza motore

def analizza_testo_ed_estrai_entita(nlp, testo: str):
    """
    Qui avviene la magia: trasformiamo una stringa di testo in dati strutturati.
    """
    # ESECUZIONE DELLA PIPELINE: 
    # nlp(testo) esegue in sequenza: tokenizzazione, tagging, parsing e infine NER.
    # L'oggetto 'doc' risultante è molto più di una stringa: sa TUTTO sulla grammatica e le entità.
    doc = nlp(testo)
    
    print("\n" + "="*60)
    print(" [REPORT] ENTITÀ IDENTIFICATE DAL MODELLO")
    print("="*60)
    
    # Controlliamo se sono state trovate entità (doc.ents contiene gli oggetti Span identificati)
    if not doc.ents:
        print("Nessuna entità trovata nel testo fornito.")
    else:
        # Cicliamo su ogni entità trovata nel documento
        for ent in doc.ents:
            # spacy.explain() ci dà una descrizione umana della categoria (es. 'PER' -> 'Person')
            spiegazione = spacy.explain(ent.label_)
            # Stampiamo il testo dell'entità, la sua etichetta tecnica e la spiegazione
            print(f"-> ENTITÀ: {ent.text:20} | CATEGORIA: {ent.label_:8} | DETTAGLIO: {spiegazione}")
    
    return doc # Restituiamo il doc elaborato per le fasi successive

def decodifica_logica_bio(doc):
    """
    Mostriamo come il computer 'etichetta' ogni singola parola internamente.
    NER usa il formato BIO: B=Begin (Inizio), I=Inside (Dentro), O=Outside (Fuori).
    """
    print("\n" + "="*60)
    print(" [DEBUG] ANALISI TECNICA: IL SISTEMA BIO")
    print("="*60)
    # Intestazione della tabella per la console
    print(f"{'PAROLA (TOKEN)':20} | {'TAG BIO':10} | {'SIGNIFICATO LOGICO'}")
    print("-" * 60)
    
    # Iteriamo su ogni singolo token (parola/punteggiatura) presente nel documento
    for token in doc:
        # token.ent_iob_ estrae il tag BIO (B, I oppure O)
        tag = token.ent_iob_
        
        # Traduciamo il tag in una descrizione comprensibile
        if tag == "B":
            stato = "INIZIO DI UN'ENTITÀ"   # Prima parola di un nome (es. 'Elon')
        elif tag == "I":
            stato = "PARTE DELL'ENTITÀ"    # Parola successiva (es. 'Musk')
        else:
            stato = "NON È UN'ENTITÀ"      # Parole comuni o punteggiatura
            
        # Stampiamo la riga corrispondente al token corrente
        print(f"{token.text:20} | {tag:10} | {stato}")

def genera_pagina_risultati(doc):
    """
    Crea una rappresentazione visiva (colorata) del testo e delle entità.
    """
    print("\n" + "="*60)
    print(" [VISUALIZZAZIONE] GENERAZIONE REPORT GRAFICO")
    print("="*60)
    
    # displacy.render trasforma il doc in codice HTML pronto per essere visualizzato
    # style="ent" indica che vogliamo evidenziare le entità nominate
    # page=True genera un documento HTML completo (col tag <html> e <body>)
    html_content = displacy.render(doc, style="ent", page=True)
    
    # Nome del file dove salveremo il report
    file_name = "risultato_analisi_ner.html"
    
    # Salviamo la stringa HTML in un file fisico sul computer
    # 'utf-8' assicura che caratteri speciali (come lettere accentuate) siano salvati correttamente
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Otteniamo il percorso completo del file appena creato
    abs_path = os.path.abspath(file_name)
    print(f"Report grafico generato con successo!")
    print(f"Puoi trovarlo qui: {abs_path}")
    
    # Comandiamo al sistema operativo di aprire il file HTML con il browser predefinito
    # Usiamo il prefisso 'file://' per indicare che è un file locale, non un sito web
    webbrowser.open(f"file://{abs_path}")

def main():
    """
    Questa è la funzione principale che orchestra l'intero processo.
    """
    # FASE 1: Preparazione - Carichiamo il cervello dell'IA
    # Usiamo 'it_core_news_lg' che è il modello più accurato per l'italiano
    nlp = carica_modello_nlp("it_core_news_lg")
    
    # FASE 2: Definizione del testo - Cosa vogliamo analizzare?
    testo_per_test = (
        "Mentre Apple annuncia nuovi uffici a Washington, "
        "Elon Musk vola in Italia per discutere con il Governo di Tesla. "
        "La Banca Centrale Europea osserva la situazione da Francoforte."
    )

    # FASE 3: Elaborazione - Passiamo il testo attraverso la pipeline di spaCy
    # Qui avviene il riconoscimento vero e proprio delle entità
    doc_elaborato = analizza_testo_ed_estrai_entita(nlp, testo_per_test)

    # FASE 4: Approfondimento - Vediamo i tag BIO dietro le quinte
    # Utile per capire come l'algoritmo separa i nomi comuni dalle entità
    decodifica_logica_bio(doc_elaborato)

    # FASE 5: Report Finale - Generiamo il file HTML interattivo
    # Apre automaticamente il browser per mostrare i risultati evidenziati
    genera_pagina_risultati(doc_elaborato)

# Questo blocco assicura che il codice parta solo se eseguiamo direttamente questo file
if __name__ == "__main__":
    main()