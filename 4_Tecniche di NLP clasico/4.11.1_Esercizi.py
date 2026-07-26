"""
================================================================================
ANALISI DI COMUNICATI FINANZIARI
================================================================================
"""

import spacy
from spacy import displacy
import os
import webbrowser
import sys

def carica_modello_nlp(nome_modello: str = "it_core_news_lg"):
    """
    Carica il modello linguistico e aggiunge regole personalizzate 
    per migliorare il riconoscimento delle date.
    """
    try:
        nlp = spacy.load(nome_modello)
        
        # --- AGGIUNTA REGOLE PERSONALIZZATE PER LE DATE ---
        mesi = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", 
                "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
        
        # Configurazione: overwrite_ents=True assicura che le nostre regole vincano
        if "entity_ruler" not in nlp.pipe_names:
            config = {"overwrite_ents": True}
            ruler = nlp.add_pipe("entity_ruler", config=config, before="ner")
            
            patterns = [
                {"label": "DATE", "pattern": [
                    {"IS_DIGIT": True}, 
                    {"LOWER": {"IN": mesi}}, 
                    {"IS_DIGIT": True}
                ]},
                {"label": "DATE", "pattern": [
                    {"LOWER": {"IN": mesi}}, 
                    {"IS_DIGIT": True}
                ]},
                {"label": "DATE", "pattern": [
                    {"IS_DIGIT": True}, 
                    {"LOWER": {"IN": mesi}}
                ]}
            ]
            ruler.add_patterns(patterns)
            
        print(f"[SISTEMA] Modello '{nome_modello}' caricato con regole DATE potenziate.")
        return nlp
    except OSError:
        print(f"[ERRORE] Modello '{nome_modello}' non trovato.")
        print(f"Esegui: python -m spacy download {nome_modello}")
        sys.exit(1)

def estrai_org_date(nlp, testo: str):
    """
    Analizza il testo e restituisce un dizionario con ORG e DATE.
    """
    doc = nlp(testo)
    
    # Debug: Stampa tutte le entità trovate prima del filtraggio
    print(f"\n[DEBUG] Entità grezze trovate: {[(ent.text, ent.label_) for ent in doc.ents]}")
    
    risultato = {
        "ORG": [],
        "DATE": []
    }
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            risultato["ORG"].append(ent.text)
        elif ent.label_ == "DATE":
            risultato["DATE"].append(ent.text)
            
    return risultato, doc

def visualizza_gpe(doc):
    """
    Genera un report HTML.
    NOTA: Abbiamo esteso la visualizzazione per includere anche le DATE
    così puoi verificare visivamente che vengano riconosciute correttamente.
    """
    print("\n" + "="*60)
    print(" [VISUALIZZAZIONE] GENERAZIONE REPORT COMPLETO (LUOGHI + DATE)")
    print("="*60)
    
    # Debug in console
    print("Entità che verranno evidenziate:")
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "DATE"]:
            print(f"- {ent.text} -> {ent.label_}")

    # Colori premium per tutte le entità che ci interessano
    colors = {
        "GPE": "linear-gradient(90deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%)",   # Rosa/Rosso
        "LOC": "linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%)",               # Verde/Azzurro
        "DATE": "linear-gradient(90deg, #fccb90 0%, #d57eeb 100%)"                # Arancio/Viola
    }
    
    # Aggiungiamo DATE al filtro per vederla nel grafico
    opzioni = {
        "ents": ["GPE", "LOC", "DATE"],
        "colors": colors
    }
    
    html_content = displacy.render(doc, style="ent", options=opzioni, page=True)
    
    file_name = "report_ner_finanziario.html"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    abs_path = os.path.abspath(file_name)
    print(f"\nReport grafico generato con successo!")
    print(f"Percorso: {abs_path}")
    
    webbrowser.open(f"file://{abs_path}")

def main():
    # 1. Carica il modello
    nlp = carica_modello_nlp("it_core_news_lg")
    
    # 2. Testo da analizzare
    frase_test = "Eni ha firmato un accordo con il Governo Egiziano il 12 Dicembre 2025 per la fornitura di gas naturale a Milano."
    
    # 3. Estrazione ORG e DATE
    entita_estratte, doc = estrai_org_date(nlp, frase_test)
    
    print("\n" + "="*60)
    print(" [RISULTATO ESTRAZIONE]")
    print("="*60)
    print(f"ORG:  {entita_estratte['ORG']}")
    print(f"DATE: {entita_estratte['DATE']}")
    
    # 4. Visualizzazione Sfida: Solo GPE
    visualizza_gpe(doc)

if __name__ == "__main__":
    main()