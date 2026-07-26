"""
=============================================================================
         ANALISI DOCUMENTALE AVANZATA: IL BARICENTRO DEL SIGNIFICATO
=============================================================================

Questo script è un framework per la gestione e il confronto di interi DOCUMENTI (oggetti `spacy.tokens.Doc`)
invece di singole parole. Nel machine learning moderno, questo approccio è alla base 
dei sistemi RAG (Retrieval-Augmented Generation) e della Document Intelligence.

ARCHITETTURA DEL CODICE:
1. LOGICA DI ESTRAZIONE: La classe `Doc` interagisce con i suoi `Token` per estrarre vettori densi 300D.
2. FILTRAGGIO SEMANTICO: Pulizia del rumore (stop-words/punteggiatura) per isolare il significato puro.
3. MOTORE DI RICERCA: Interazione tra `nlp.pipe` (efficienza) e `scipy.spatial` (geometria).
4. ANALISI DEI LIMITI: Dimostrazione di come i vettori statici (Word2Vec/GloVe) falliscono nel contesto.

"""

import spacy                # Core NLP: gestisce la pipeline di analisi e i modelli linguistici
import numpy as np          # Matematica vettoriale: usato per aggregare i vettori (medie e baricentri)
from scipy import spatial   # Geometria spaziale: fornisce l'algoritmo per calcolare la distanza coseno

def get_cleaned_vector(doc):
    """
    Calcola il vettore rappresentativo di un documento (Doc) pulendolo dal 'rumore'.
    
    INTERAZIONE FRA CLASSI:
    - Riceve un oggetto `Doc` (un'intera frase o paragrafo).
    - Itera su ogni oggetto `Token` all'interno del `Doc`.
    - Accede alle proprietà booleane del Token (`is_stop`, `is_punct`) per decidere cosa scartare.
    - Estrae l'attributo `token.vector` (un array NumPy 300D) dai termini validi.
    """
    
    # List Comprehension: Filtriamo i token in base a criteri di rilevanza semantica
    # 1. token.has_vector: Verifica se il modello ha un embedding associato (evita OOV - Out Of Vocabulary)
    # 2. not token.is_stop: Esclude congiunzioni, articoli, preposizioni (rumore statistico)
    # 3. not token.is_punct: Esclude virgole, punti, ecc. che non hanno peso semantico
    vectors = [token.vector for token in doc if token.has_vector and not token.is_stop and not token.is_punct]
    
    if not vectors:
        # Edge Case: Se il testo è vuoto o fatto solo di stop-words (es. "che il lo"), 
        # restituiamo un vettore nullo della dimensione corretta (300 per il modello _lg)
        return np.zeros((300,))
    
    # OPERAZIONE MATEMATICA: Calcolo del Centroide
    # Sommiamo tutti i vettori e dividiamo per il numero di parole rilevanti.
    # axis=0 assicura che la media venga fatta componente per componente (300 dimensioni)
    return np.mean(vectors, axis=0)

def motore_di_ricerca_semantica(nlp, query, documenti):
    """
    Esegue una ricerca basata sul concetto (vettore), non sulla coincidenza di caratteri.
    
    PARAMETRI:
    - nlp: L'oggetto Language caricato (es. it_core_news_lg).
    - query: La stringa cercata dall'utente.
    - documenti: Una lista di stringhe che compongono il nostro database di conoscenza.
    """
    print(f"\n--- [LOG]: AVVIO RICERCA SEMANTICA ---")
    print(f"Query Digitata: '{query}'")
    
    # 1. TRASFORMAZIONE QUERY: Convertiamo la query in un vettore pulito
    # nlp(query) crea un oggetto Doc, che passiamo alla nostra funzione di pulizia
    query_vector = get_cleaned_vector(nlp(query))
    
    risultati = []
    
    # 2. OTTIMIZZAZIONE SCALABILE: nlp.pipe
    # Invece di chiamare nlp(doc) in un loop (lento), usiamo nlp.pipe:
    # - Elabora i testi in batch (gruppi).
    # - Sfrutta il multithreading e le ottimizzazioni C di spaCy.
    # - Restituisce un iteratore di oggetti Doc.
    for doc in nlp.pipe(documenti):
        # Per ogni documento nel database, calcoliamo il suo baricentro semantico
        doc_vector = get_cleaned_vector(doc)
        
        # 3. CALCOLO SIMILARITÀ (1 - Distanza Coseno)
        # La distanza coseno misura l'angolo tra i due vettori nello spazio 300D.
        # Se l'angolo è 0, i vettori sono sovrapposti (uguaglianza semantica).
        # Convertiamo la distanza (0 a 2) in uno score (0 a 1) dove 1 è il massimo.
        score = 1 - spatial.distance.cosine(query_vector, doc_vector)
        risultati.append((doc.text, score))
    
    # 4. RANKING: Ordiniamo i risultati dal più simile al meno simile
    risultati.sort(key=lambda x: x[1], reverse=True)
    
    # Output dei risultati formattati
    for testo, score in risultati:
        status = "MATCH" if score > 0.6 else "RELEVANT" if score > 0.4 else "LOW"
        print(f"[{score:.4f} - {status}] {testo[:70]}...")

def analisi_contrasto_documentale(nlp, doc_a, doc_b):
    """
    Confronta due documenti evidenziando la differenza tra approccio standard e raffinato.
    """
    # .similarity() è il metodo integrato di spaCy: fa la media Semplice di TUTTE le parole
    similarity = doc_a.similarity(doc_b)
    
    print(f"\n--- [DEBUG]: ANALYSIS COMPARISON ---")
    print(f"A: '{doc_a.text[:40]}...'")
    print(f"B: '{doc_b.text[:40]}...'")
    
    # Mostriamo il vantaggio del filtraggio stop-words
    vec_a = get_cleaned_vector(doc_a)
    vec_b = get_cleaned_vector(doc_b)
    clean_sim = 1 - spatial.distance.cosine(vec_a, vec_b)
    
    print(f"Similarità Standard (Include rumore): {similarity:.4f}")
    print(f"Similarità Raffinata (Focus concetti): {clean_sim:.4f}")

def main():
    # SETUP: Carichiamo il modello Large. Contiene 500.000+ vettori unici.
    # È indispensabile per analisi di similarità; il modello 'sm' non ha vettori reali.
    try:
        nlp = spacy.load("it_core_news_lg")
    except OSError:
        print("Modello non trovato. Eseguire: python -m spacy download it_core_news_lg")
        return

    # DATABASE: Il nostro piccolo 'Knowledge Graph' testuale
    conoscenze = [
        "Il motore a scoppio utilizza combustibili fossili per generare energia meccanica.",
        "Le energie rinnovabili come il solare e l'eolico sono il futuro del pianeta.",
        "La cucina mediterranea si basa su olio d'oliva, verdure fresche e cereali.",
        "Il machine learning permette ai computer di apprendere dai dati senza programmazione esplicita.",
        "Investire in borsa richiede una profonda conoscenza dei mercati finanziari."
    ]

    # TEST 1: Ricerca di un concetto non presente testualmente
    # Cerchiamo "veicoli a motore". Notate come troverà il documento sui motori a scoppio
    # anche se sono parole diverse: questo è il potere del Word Embedding.
    motore_di_ricerca_semantica(nlp, "veicoli a motore", conoscenze)
    
    # TEST 2: Concetti astratti
    motore_di_ricerca_semantica(nlp, "intelligenza artificiale", conoscenze)

    # TEST 3: Limiti dei vettori statici (Il problema "NON")
    # I Word Embeddings classici sono una 'zuppa di parole': l'ordine non conta.
    doc_neg = nlp("Il film non era affatto bello, mi sono annoiato.")
    doc_pos = nlp("Il film era molto bello, non mi sono annoiato.")
    
    print("\n--- [WARNING]: IL LIMITE DEI VETTORI STATICI ---")
    print(f"Sim. Semantica: {doc_neg.similarity(doc_pos):.4f}")
    print("ANALISI: Poiché contengono quasi le stesse parole, il baricentro è quasi identico.")
    print("SOLUZIONE: Per distinguere 'Bello' da 'Non Bello' servono i Transformer (BERT/GPT).")

    doc1 = nlp("Il sistema di propulsione richiede carburante.")
    doc2 = nlp("Un motore ha bisogno di benzina per funzionare.")
    
    # Ora la chiamiamo davvero!
    analisi_contrasto_documentale(nlp, doc1, doc2)

if __name__ == "__main__":
    main()

# =============================================================================
#  NOTE TECNICHE PER LO STUDENTE
# =============================================================================
# 1. IL DOC (DOCUMENTO) COME VETTORE:
#    In spaCy, doc.vector è la media aritmetica dei vettori dei token componenti.
#    È come se prendessimo le coordinate GPS di 10 persone in una piazza e 
#    calcolassimo il punto medio: quel punto rappresenta la "posizione del gruppo".
#
# 2. PERCHÉ PULIRE LE STOP-WORDS?
#    Le parole come "il", "di", "per" sono ovunque. Matematicamente, hanno vettori
#    che 'tirano' ogni frase verso il centro dello spazio vettoriale. 
#    Togliendole, permettiamo alle parole chiave (es. "motore", "cucina") di 
#    esprimere la loro vera direzione senza interferenze.
#
# 3. PERFORMANCE CON nlp.pipe:
#    In produzione, non iterare mai manualmente con un for se hai più di 100 testi.
#    nlp.pipe è thread-safe e scritto in Cython. Gestisce la memoria in modo 
#    estremamente efficiente, evitando l'overhead del garbage collector di Python.
# =============================================================================
