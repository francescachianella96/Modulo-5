"""
Classificazione con TF-IDF + Logistic Regression
Questo script illustra l'implementazione della Regressione Logistica utilizzando 
scikit-learn, focalizzandosi sull'interpretabilità dei coefficienti e sulla regolarizzazione.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def get_educational_dataset():
    """
    Crea un dataset di esempio per illustrare il sentiment analysis.
    
    Returns:
        tuple: (testi, etichette)
    """
    texts = [
        "Il film è stato assolutamente fantastico ed eccellente",
        "Pessima esperienza, tutto molto brutto e noioso",
        "Lo raccomando a tutti, una visione incredibile",
        "Non mi è piaciuto per niente, uno spreco di tempo",
        "La trama era geniale e gli attori bravissimi",
        "Terribile, non guardatelo, storia pessima",
        "Un capolavoro del cinema moderno",
        "Disastro totale, sceneggiatura imbarazzante"
    ]
    # 1 = Positivo, 0 = Negativo
    labels = [1, 0, 1, 0, 1, 0, 1, 0]
    return texts, labels

def analyze_model_weights(vectorizer, model):
    """
    Estrae e visualizza i coefficienti del modello per spiegare l'interpretabilità.
    
    Args:
        vectorizer: Il TfidfVectorizer addestrato.
        model: Il modello di LogisticRegression addestrato.
    """
    # Otteniamo i nomi delle feature (le parole nel vocabolario)
    feature_names = vectorizer.get_feature_names_out()
    # Otteniamo i coefficienti (i pesi assegnati a ogni parola)
    coefficients = model.coef_[0]
    
    # Creiamo una lista di coppie (parola, peso)
    feature_importance = list(zip(feature_names, coefficients))
    
    # Ordiniamo per peso crescente
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- ANALISI DEI COEFFICIENTI (Interpretabilità) ---")
    print("Top 3 parole che spingono verso il POSITIVO (Coefficienti +):")
    for word, weight in feature_importance[:3]:
        print(f" -> {word}: {weight:.4f}")
        
    print("\nTop 3 parole che spingono verso il NEGATIVO (Coefficienti -):")
    for word, weight in feature_importance[-3:]:
        # Le parole negative avranno i pesi più bassi (più negativi)
        print(f" -> {word}: {weight:.4f}")

def main():
    # 1. Caricamento Dati
    texts, labels = get_educational_dataset()

    # 2. Vettorizzazione TF-IDF (Slide 5)
    # Trasforma il testo in una matrice numerica pesata
    # min_df=1 permette di includere anche parole che appaiono una sola volta (ideale per piccoli dataset)
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # 3. Creazione del Modello con Regolarizzazione (Slide 11-12)
    # 'C' è l'inverso della forza di regolarizzazione. 
    # Un valore basso (es. 0.1) = Regolarizzazione forte (modello semplice)
    # Un valore alto (es. 10.0) = Regolarizzazione debole (modello complesso)
    model = LogisticRegression(
        penalty='l2',    # Tipo di penalità (Ridge)
        C=1.0,           # Bilanciamento Bias-Varianza
        solver='liblinear'
    )

    # 4. Addestramento (Slide 6)
    model.fit(X, y)
    print("Modello addestrato con successo.")

    # 5. Analisi delle Feature (Slide 7-10)
    analyze_model_weights(vectorizer, model)

    # 6. Esempio di Predizione (Inferenza)
    test_reviews = [
        "Un film incredibile e fantastico",
        "Esperienza pessima e noiosa"
    ]
    
    # Trasformiamo i nuovi testi usando il vettorizzatore già addestrato
    X_test = vectorizer.transform(test_reviews)
    
    # Prediciamo le probabilità (Slide 4 - La funzione Sigmoide)
    probabilities = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    print("\n--- TEST DI INFERENZA ---")
    for i, review in enumerate(test_reviews):
        class_label = "Positivo" if predictions[i] == 1 else "Negativo"
        prob_pos = probabilities[i][1]
        print(f"Testo: '{review}'")
        print(f"Probabilità Positivo: {prob_pos:.2%} -> Risultato: {class_label}\n")

if __name__ == "__main__":
    main()