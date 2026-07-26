import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def genera_dati_didattici():
    """
    DATA GENERATION: Creazione di un dataset VOLUTAMENTE sbilanciato.
    
    Minoranza (Positivi): 5 esempi
    Maggioranza (Negativi): 15 esempi
    Rapporto 1:3
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
        "Recitazione da brividi, ma nel senso sbagliato del termine",
        "Film pessimo, non lo consiglio a nessuno",
        "Una perdita di tempo incredibile",
        "Storia senza capo né coda, davvero brutto"
    ]
    
    # Etichette: 5 positivi (1), 15 negativi (0)
    labels = np.array([1]*5 + [0]*15)
    
    return recensioni, labels

def addestra_svm(X, y, bilancia=False, C=1.0):
    """
    Addestra il modello LinearSVC con parametri specifici.
    C: Inverse of regularization strength. Smaller values specify stronger regularization.
    """
    weight_strategy = 'balanced' if bilancia else None
    model = LinearSVC(C=C, class_weight=weight_strategy, random_state=42, max_iter=10000)
    model.fit(X, y)
    return model

def confronta_e_visualizza(modelli, nomi_scenari, feature_names, top_n=8):
    """
    Visualizza i coefficienti di più modelli per confrontare l'impatto dei parametri.
    """
    num_modelli = len(modelli)
    fig, axes = plt.subplots(1, num_modelli, figsize=(18, 10), sharey=False)
    
    if num_modelli == 1:
        axes = [axes]

    for i, (model, nome) in enumerate(zip(modelli, nomi_scenari)):
        coef = model.coef_.ravel()
        
        # Selezioniamo i top N positivi e negativi
        top_positive_indices = np.argsort(coef)[-top_n:]
        top_negative_indices = np.argsort(coef)[:top_n]
        indices = np.concatenate([top_negative_indices, top_positive_indices])
        
        parole = [feature_names[idx] for idx in indices]
        pesi = coef[indices]
        
        colori = ['#e74c3c' if c < 0 else '#2ecc71' for c in pesi]
        axes[i].barh(parole, pesi, color=colori)
        axes[i].axvline(0, color='black', linewidth=0.8)
        axes[i].set_title(f"Scenario: {nome}", fontsize=12)
        axes[i].grid(axis='x', alpha=0.3)
        
    plt.tight_layout()
    plt.suptitle("Detective dei Dati: Impatto del Bilanciamento e della Regolarizzazione", fontsize=16, y=1.02)
    plt.show()

def main():
    # 1. Preparazione Dati
    testi, labels = genera_dati_didattici()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(testi)
    vocab = vectorizer.get_feature_names_out()

    print("--- Analisi Sperimentale SVM ---")
    
    # SCENARIO 1: Sbilanciato (Default)
    # Spesso ignora la classe minoritaria o dà pesi bassi alle sue parole chiave.
    svm_sbilanciata = addestra_svm(X, labels, bilancia=False, C=1.0)
    
    # SCENARIO 2: Bilanciato
    # Attiva l'iperpiano che 'ascolta' di più i pochi esempi positivi.
    svm_bilanciata = addestra_svm(X, labels, bilancia=True, C=1.0)
    
    # SCENARIO 3: Bilanciato + Alta Regolarizzazione (C piccolo)
    # L'iperpiano cerca di essere più 'largo' e robusto, 'restringendo' i coefficienti.
    svm_regolarizzata = addestra_svm(X, labels, bilancia=True, C=0.01)

    # Raccolta modelli per il confronto
    modelli = [svm_sbilanciata, svm_bilanciata, svm_regolarizzata]
    nomi = ["Sbilanciata (C=1.0)", "Bilanciata (C=1.0)", "Bilanciata + Regol. (C=0.01)"]

    # Visualizzazione
    print("\nVisualizzazione dei coefficienti in corso...")
    confronta_e_visualizza(modelli, nomi, vocab)

    # Analisi dei risultati (Testuale)
    for model, nome in zip(modelli, nomi):
        pred = model.predict(X)
        print(f"\n--- {nome} ---")
        print(classification_report(labels, pred, target_names=['Negativo', 'Positivo'], zero_division=0))

if __name__ == "__main__":
    main()