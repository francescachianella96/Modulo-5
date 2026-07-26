"""
====================================================================================================
ESERCIZIO: ANALISI AVANZATA CON HUGGING FACE TRANSFORMERS
====================================================================================================
OBIETTIVI:
1. Sentiment Analysis su una frase complessa in inglese.
2. Zero-Shot Classification in italiano per una recensione specifica.
3. Confronto dei punteggi e analisi della confidenza del modello.
====================================================================================================
"""

import torch
from transformers import pipeline

def sentiment_analysis_complessa():
    """
    Punto 1: Istanzia una pipeline di 'sentiment-analysis' e testa una frase complessa in inglese.
    Una frase complessa contiene sfumature positive e negative, testando la capacità del modello
    di cogliere il sentimento prevalente.
    """
    print("\n--- [Task 1: Sentiment Analysis su Testo Complesso] ---")
    
    # Utilizziamo un modello standard ma potente per l'inglese
    analizzatore = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
    
    # Frase complessa: inizia con una critica ma finisce con un forte apprezzamento (sentiment positivo)
    testo_complesso = (
        "While the initial configuration was somewhat intuitive and the documentation occasionally "
        "lacks clarity, the raw processing speed and the sheer efficiency of the library have "
        "completely outperformed every competitor, making it an essential tool for our tech stack."
    )
    
    risultato = analizzatore(testo_complesso)[0]
    
    print(f"Testo Inglese: {testo_complesso}")
    print(f"Risultato: {risultato['label']} | Punteggio (Confidenza): {risultato['score']:.4f}")
    return risultato

def zero_shot_italiano():
    """
    Punto 2 & 3: Utilizza una pipeline di 'zero-shot-classification' per classificare una recensione 
    italiana in tre categorie: 'Entusiasmo', 'Delusione', 'Richiesta di supporto'.
    Per l'italiano utilizziamo un modello multilingue affidabile (mDeBERTa-v3).
    """
    print("\n--- [Task 2: Zero-Shot Classification in Italiano] ---")
    
    # Modello consigliato per lo zero-shot multilingue di alta qualità
    modello_zeroshot = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    classificatore = pipeline("zero-shot-classification", model=modello_zeroshot, framework="pt")
    
    # Recensione italiana con sentiment negativo e richiesta d'aiuto
    recensione = "Il pacco è arrivato in ritardo e la scatola era visibilmente danneggiata. Sono molto scontento dell'acquisto, come posso chiedere il rimborso?"
    
    # Categorie richieste dall'esercizio
    categorie = ["Entusiasmo", "Delusione", "Richiesta di supporto"]
    
    # Il parametro multi_label=True permetterebbe di assegnare più etichette, 
    # ma qui seguiamo la classificazione standard per vedere i pesi relativi.
    risultato = classificatore(recensione, candidate_labels=categorie)
    
    print(f"Testo Italiano: {recensione}")
    for label, score in zip(risultato['labels'], risultato['scores']):
        print(f"  -> Categoria: {label:22} | Punteggio: {score:.4f}")
    
    return risultato

def confronta_risultati(res_sentiment, res_zeroshot):
    """
    Punto 3: Confronta i punteggi ottenuti.
    """
    print("\n--- [Task 3: Confronto e Analisi dei Punteggi] ---")
    
    # Nel sentiment analysis binario, il punteggio è la probabilità della classe predetta.
    print(f"1. Confidenza Sentiment (Inglese): {res_sentiment['score']:.4f}")
    
    # Nello zero-shot, abbiamo una distribuzione di probabilità sulle tre classi.
    # Prendiamo la classe con il punteggio più alto.
    top_label = res_zeroshot['labels'][0]
    top_score = res_zeroshot['scores'][0]
    print(f"2. Confidenza Dominante Zero-Shot (Italiano - {top_label}): {top_score:.4f}")
    
    print("\nOsservazioni:")
    if res_sentiment['score'] > 0.9:
        print("- Il modello di sentiment è estremamente sicuro del risultato nonostante la complessità sintattica.")
    
    if top_score > 0.5:
        print(f"- Il modello zero-shot ha identificato correttamente '{top_label}' come categoria principale.")
    else:
        print("- Il modello zero-shot mostra incertezza, probabilmente il testo appartiene a più categorie contemporaneamente.")

if __name__ == "__main__":
    # Esecuzione del flusso
    res_s = sentiment_analysis_complessa()
    res_z = zero_shot_italiano()
    confronta_risultati(res_s, res_z)