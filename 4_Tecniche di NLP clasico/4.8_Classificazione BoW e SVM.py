"""

SISTEMA: Sentiment Analysis con SVM Lineare (Pure Scikit-Learn)

ARCHITETTURA DEL CODICE:
1. DATA GENERATION: Creiamo testi grezzi (Input umano).
2. VECTORIZATION (Ponte): Trasformiamo il testo in coordinate numeriche (Input macchina).
3. MODELING (Il Cuore): L'SVM traccia l'iperpiano che separa i territori.
4. ANALYTICS (L'Interprete): Estraiamo i pesi per capire la logica del modello.

RELAZIONE TRA PARTI:
- Il 'vectorizer' definisce le dimensioni dello spazio in cui la 'svm' lavorerà.
- La 'svm' genera coefficienti che tornano al 'vectorizer' per essere associati alle parole.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def genera_dati_didattici():
    """
    PUNTO DI PARTENZA: Creazione del Dataset.
    
    Relazione: Fornisce le stringhe di testo che il 'CountVectorizer' dovrà mappare.
    Creiamo volontariamente più negativi che positivi per testare la 'giustizia statistica'.
    """
    recensioni = [
        # --- CLASSE POSITIVA (1) - La minoranza ---
        "Un capolavoro assoluto, sceneggiatura e attori brillanti",
        "Emozionante, profondo e visivamente splendido",
        "Una delle migliori esperienze cinematografiche di sempre",
        "Regia magistrale e fotografia da togliere il fiato",
        "Un viaggio incredibile nei sentimenti umani, consigliatissimo",
        
        # --- CLASSE NEGATIVA (0) - La maggioranza ---
        "Noioso, prevedibile e recitato male",
        "Uno spreco di soldi, assolutamente sconsigliato",
        "Pessimo montaggio, trama inesistente e banale",
        "Terribile, non vedevo l'ora che finisse",
        "Delusione totale, regia mediocre",
        "Brutto, lento e senza alcun senso logico",
        "Sceneggiatura imbarazzante, un insulto all'intelligenza",
        "Attori senza carisma in una storia piatta e vuota",
        "Effetti speciali ridicoli, soldi buttati",
        "Non succede nulla per due ore, una tortura infinita",
        "Confuso, pretenzioso e inutilmente lungo",
        "Recitazione da brividi, ma nel senso sbagliato del termine"
    ]
    
    # Etichette corrispondenti (5 '1' e 12 '0')
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    return recensioni, labels

def addestra_svm_interpretabile(X, y, bilancia=True):
    """
    IL CERVELLO GEOMETRICO: Addestramento del modello.
    
    Interazione: Riceve la matrice X (coordinate) e cerca l'iperpiano di separazione.
    Slide 4 & 6: Massimizza il margine (la distanza dai Support Vectors).

    Parametri critici:
    - C (Regolarizzazione): Controlla quanto l'iperpiano deve essere 'rigido' o 'elastico'.
    - class_weight (Bilanciamento): Se attivo, dà più forza ai pochi esempi positivi (Slide 8).
    """
    # Se 'balanced', la SVM dà un 'megafono' alla classe con meno esempi (Slide 10)
    weight_strategy = 'balanced' if bilancia else None
    
    # LinearSVC è l'implementazione matematica pura dell'iperpiano lineare.
    model = LinearSVC(C=1.0, class_weight=weight_strategy, random_state=42)
    
    # .fit() è il processo in cui l'elastico matematico si tende tra i punti per trovare l'equilibrio.
    model.fit(X, y)
    return model

def visualizza_coefficienti(model, feature_names, top_n=5):
    """
    L'INTERPRETE: Traduzione della geometria in linguaggio umano.
    
    Interazione: Prende i pesi (W) dalla SVM e i nomi delle parole dal Vectorizer.
    Slide 11-14: Senza questa funzione, la SVM sarebbe una 'scatola nera'.
    """
    # .coef_ contiene i pesi: un peso positivo 'tira' verso la classe 1, uno negativo verso la 0.
    coef = model.coef_.ravel()
    
    # np.argsort() crea una classifica: dai termini più 'negativi' ai più 'positivi' (Slide 14).
    top_positive_indices = np.argsort(coef)[-top_n:]
    top_negative_indices = np.argsort(coef)[:top_n]
    
    # Uniamo gli indici per avere una visione d'insieme dei due poli semantici.
    interesting_indices = np.concatenate([top_negative_indices, top_positive_indices])
    
    # Ripeschiamo i nomi delle parole corrispondenti agli indici (Ponte tra numeri e testo).
    parole = [feature_names[i] for i in interesting_indices]
    pesi = coef[interesting_indices]
    
    # Rendering grafico: visualizziamo la 'forza' di ogni parola nell'influenzare il modello.
    plt.figure(figsize=(12, 7))
    colori = ['#e74c3c' if c < 0 else '#2ecc71' for c in pesi] 
    plt.barh(parole, pesi, color=colori)
    plt.axvline(0, color='black', linewidth=0.8) # Il confine dello zero (l'iperpiano neutro)
    plt.title("Analisi dei Coefficienti SVM: La Forza Semantica delle Parole", fontsize=14)
    plt.xlabel("Peso Numerico (Impatto sulla Classificazione)", fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.show()

def main():
    """
    ORCHESTRATORE: Gestisce il flusso dell'informazione.
    """
    # STEP 1: Acquisizione dati
    testi, labels = genera_dati_didattici()
    
    # STEP 2: Trasformazione Geometrica (Bag of Words)
    # Questa classe mappa ogni parola a una colonna specifica: crea lo 'spazio' (Slide 3).
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(testi) # X è ora una matrice di coordinate.
    vocab = vectorizer.get_feature_names_out() # Salviamo i nomi per l'interpretazione finale.
    
    # STEP 3: Modellazione (Training)
    # Passiamo le coordinate e le etichette alla SVM.
    print("--- Addestramento SVM: Ricerca dell'Iperpiano Ottimale ---")
    svm_model = addestra_svm_interpretabile(X, labels, bilancia=True)
    
    # STEP 4: Validazione (Prediction)
    # Chiediamo al modello di classificare i dati per vedere quanto è diventato accurato.
    predizioni = svm_model.predict(X)
    print("\nReport Qualitativo (Accuracy vs Precision/Recall):")
    print(classification_report(labels, predizioni, target_names=['Negativo', 'Positivo']))
    
    # STEP 5: Spiegabilità (Visualizzazione)
    # Colleghiamo i risultati del modello (svm_model) con il dizionario creato all'inizio (vocab).
    print("\nGenerazione del grafico dei poli semantici...")
    visualizza_coefficienti(svm_model, vocab)

if __name__ == "__main__":
    main()