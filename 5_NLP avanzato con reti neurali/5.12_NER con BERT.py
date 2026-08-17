"""
====================================================================================================
SFIDA NER: BIO-BERT (TRANSFORMERS) VS SPACY (MODELLI STATISTICI)
====================================================================================================
In questo script mettiamo a confronto due generazioni di Intelligenza Artificiale:
1. spaCy: Rappresenta l'approccio classico statistico (veloce, ma limitato dal contesto locale).
2. BioBERT: Rappresenta lo stato dell'arte dei Transformers (lento, ma con una comprensione 
   profonda e semantica del dominio medico).

Obiettivo: Dimostrare come BERT riesca a "leggere tra le righe" nei settori specialistici.
====================================================================================================
"""

import os

# --- CONFIGURAZIONE BACKEND ---
# Keras 3 è agnostico: qui forziamo l'uso di PyTorch come "motore" di calcolo.
# Questa riga deve essere eseguita PRIMA di importare keras o altri framework.
os.environ["KERAS_BACKEND"] = "torch"

import spacy
from transformers import pipeline

def load_models():
    """
    Inizializza e carica in memoria i due cervelli artificiali che confronteremo.
    
    Interazione:
    - nlp_spacy: Carica pesi statistici pre-calcolati per la lingua inglese generica.
    - ner_biobert: Scarica (se non presente) un modello BERT mastodontico addestrato su PubMed.
    """
    print("\n[INFO] Caricamento 'cervelli' in corso...")
    
    # --- SFIDANTE 1: spaCy (Tradizione) ---
    try:
        # Carica il modello statistico "large" per l'inglese.
        # È basato su una pipeline di regole e pesi lineari molto veloci.
        nlp_spacy = spacy.load("en_core_web_lg")
        print("[OK] spaCy caricato (Modello Statistico).")
    except:
        # Se il modello non è installato (es. manca il download), gestiamo l'errore.
        print("[!] Attenzione: Modello spaCy 'en_core_web_sm' non trovato.")
        nlp_spacy = None

    # --- SFIDANTE 2: BioBERT (Innovazione) ---
    # Il nome del modello su HuggingFace: una versione di BERT specializzata in medicina.
    model_name = "d4data/biomedical-ner-all"
    
    # La 'pipeline' è l'astrazione più alta possibile:
    # 1. Carica il Tokenizer (per spezzare le parole in numeri).
    # 2. Carica il Modello (i miliardi di parametri di BERT).
    # 3. Imposta l'aggregation_strategy="simple" per unire i pezzi di parole (##) in entità intere.
    # 4. framework="pt": Forza l'uso di PyTorch (fondamentale con Keras 3 installato).
    ner_biobert = pipeline("ner", model=model_name, aggregation_strategy="simple", framework="pt")
    print("[OK] BioBERT caricato (Modello Transformer di settore).")
    
    return nlp_spacy, ner_biobert

def run_comparison(text):
    """
    Funzione principale che mette i due modelli davanti allo stesso testo specialistico.
    
    Parametri:
    - text: La frase medica complessa da analizzare.
    """
    # 1. Otteniamo i modelli pronti all'uso
    nlp_spacy, ner_biobert = load_models()
    
    print(f"\n{'='*70}")
    print(f"TESTO DA ANALIZZARE:\n'{text}'")
    print(f"{'='*70}")

    # --- FASE 1: ANALISI CON SPACY ---
    print("\n>>> ANALISI CON SPACY (Modello Statistico Generalista):")
    if nlp_spacy:
        # spaCy processa il testo in un colpo solo creando un oggetto 'Doc'
        doc = nlp_spacy(text)
        
        # doc.ents contiene le entità che spaCy ha 'indovinato'
        if not doc.ents:
            print("  [X] Nessuna entità medica trovata. spaCy non riconosce termini tecnici.")
        for ent in doc.ents:
            # ent.text: la parola trovata | ent.label_: la categoria (es. PERSON, ORG)
            print(f"  - Trovato: {ent.text:25} | Categoria: {ent.label_}")
    else:
        print("  [!] spaCy non disponibile.")

    # --- FASE 2: ANALISI CON BIO-BERT ---
    print("\n>>> ANALISI CON BIO-BERT (Deep Learning di Settore):")
    
    # La pipeline chiamerà internamente BERT, calcolerà l'attenzione e restituirà una lista
    results = ner_biobert(text)
    
    if not results:
        print("  [X] Nessuna entità trovata.")
    for ent in results:
        # ent['word']: il termine medico identificato
        # ent['entity_group']: la classe (es. DISEASE, DRUG)
        # ent['score']: quanto BERT è sicuro della sua risposta (da 0 a 1)
        print(f"  - Trovato: {ent['word']:25} | Categoria: {ent['entity_group']:8} | Confidenza: {ent['score']:.2f}")

# --- PUNTO DI INGRESSO (Main) ---
if __name__ == "__main__":
    # Scegliamo una frase "trappola": contiene termini medici che sembrano parole comuni.
    # 'Acute Myocardial Infarction' (Infarto) è un'entità complessa e annidata.
    clinical_case = (
        "The patient showed symptoms of Acute Myocardial Infarction "
        "and was treated with Aspirin and Heparin to inhibit Platelet aggregation."
    )
    
    # Avviamo il confronto
    run_comparison(clinical_case)

# --- GUIDA ALLA LETTURA DEI RISULTATI PER STUDENTI ---
"""
PERCHÉ VEDI QUELLO CHE VEDI? (Analisi riga per riga dei concetti):

1. LA STRUTTURA DEL CODICE: Il codice è diviso in 'Caricamento' (pesante) ed 'Esecuzione' (veloce).
   In AI, caricare i modelli in memoria è l'operazione più costosa; una volta pronti, 
   possono analizzare migliaia di frasi.

2. IL "COMPORTAMENTO" DI SPACY: Noterai che spaCy potrebbe scansionare 'Aspirin' ma ignorare 
   'Platelet aggregation'. Questo perché spaCy non "capisce" il senso, cerca solo pattern 
   statistici comuni. Se una parola non era nel suo dizionario di addestramento, non esiste.

3. IL "RAGIONAMENTO" DI BERT: BioBERT invece guarda la parola 'Platelet' e vede che è vicina 
   a 'inhibit'. Grazie al meccanismo di SELF-ATTENTION (Slide 8), capisce che 'aggregation' 
   è collegata a 'Platelet' e formano un unico concetto biologico.

4. AGGREGATION STRATEGY: La riga 'aggregation_strategy="simple"' è fondamentale. 
   Poiché BERT spezza le parole lunghe (es. 'Infarction' -> 'Infar', '##ction'), 
   questa opzione dice alla pipeline: "Riappiccica i pezzi prima di mostrarmeli".

5. LE CATEGORIE: Noterai che BioBERT usa classi come 'DISEASE' o 'DRUG', mentre spaCy usa 
   'ORG' o 'GPE'. Questa è la forza della 'Domain Specialization' (Slide 12).
"""