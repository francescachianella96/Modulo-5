import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib # Utilizzato per la persistenza del modello (salvataggio su disco)

"""
PIPELINE DI ELABORAZIONE TESTUALE CON TF-IDF
-------------------------------------------
Questo script implementa una pipeline completa per trasformare testo non strutturato
in rappresentazioni numeriche pesate tramite l'algoritmo TF-IDF (Term Frequency-Inverse Document Frequency).

Interazioni principali:
1. TfidfVectorizer: Classe core che analizza il corpus, crea il dizionario e calcola i pesi.
2. DataFrame (Pandas): Utilizzato per strutturare l'output numerico (matrice sparsa -> densa) 
   rendendolo leggibile e analizzabile come una tabella.
3. Joblib: Gestisce il ciclo di vita del modello, permettendo di riutilizzarlo in produzione.
"""

# 1. PREPARAZIONE DEL CORPUS (Dataset di Input)
# Un "corpus" è semplicemente una lista di stringhe (documenti).
# L'algoritmo ha bisogno di vedere l'intero insieme per determinare quanto una parola sia "rara".
corpus = [
    "Il Deep Learning rivoluziona l'intelligenza artificiale.",
    "L'intelligenza artificiale e le reti neurali sono il futuro.",
    "Le reti neurali sono ispirate al cervello umano.",
    "Il Deep Learning richiede grandi quantità di dati."
]

# 2. CONFIGURAZIONE DEL VETTORIZZATORE TF-IDF
# Inizializziamo l'oggetto TfidfVectorizer che incapsula tutta la logica di pre-processing.
vectorizer = TfidfVectorizer(
    lowercase=True,      # Converte tutto in minuscolo per evitare differenze (es: "Il" vs "il")
    stop_words=None,     # In questo esempio includiamo tutto per studiare l'effetto dell'IDF sui connettivi
    use_idf=True,        # Abilita il calcolo del peso inverso della frequenza nei documenti
    smooth_idf=True,     # Aggiunge 1 ai conteggi (laplace smoothing) per prevenire divisioni per zero
    norm='l2'            # Normalizzazione Euclidea: rende i vettori di lunghezza 1, 
                         # fondamentale affinché documenti lunghi non abbiano pesi sproporzionati.
)

# 3. FIT E TRASFORMAZIONE (Core Logic)
# Questa riga esegue due operazioni distinte ma correlate:
# - .fit(): Scansiona il 'corpus', estrae le parole univoche (Vocabulary) e calcola l'IDF globale.
# - .transform(): Prende ogni singolo documento e lo converte in un vettore di numeri basato sul vocabolario.
# Risultato: Una 'sparse matrix' dove ogni riga è un documento e ogni colonna è una parola.
tfidf_matrix = vectorizer.fit_transform(corpus)

# 4. ANALISI E VISUALIZZAZIONE DEI RISULTATI
# Estraiamo i nomi delle colonne (le parole del dizionario ordinato alfabeticamente)
feature_names = vectorizer.get_feature_names_out()

# Poiché la matrice TF-IDF di scikit-learn è ottimizzata per la memoria (CSR format),
# la convertiamo in 'dense' (matrice standard) per caricarla in un DataFrame Pandas.
dense_matrix = tfidf_matrix.todense()
df_tfidf = pd.DataFrame(dense_matrix, columns=feature_names)

print("--- Matrice TF-IDF (Rappresentazione Tabellare) ---")
# Visualizziamo i pesi: valori alti indicano parole "caratterizzanti" per quel documento.
print(df_tfidf.head())

# 5. ISPEZIONE DELL'IDF (Importanza Semantica Globale)
# L'IDF misura la "rarità" di una parola: IDF = log(N/n) dove N è il tot documenti.
# Creiamo un dizionario mappando parola -> suo peso IDF calcolato durante il .fit()
idf_values = dict(zip(feature_names, vectorizer.idf_))

print("\n--- Analisi della Rarità (Top 5 Termini con IDF più alto) ---")
# Più alto è il valore IDF, più la parola è unica e informativa nel contesto del corpus.
for word, idf in sorted(idf_values.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Termine Tecnico/Raro: '{word}' -> Peso IDF: {idf:.4f}")

# 6. PERSISTENZA DEL MODELLO (Best Practice per Produzione)
# Non salviamo i dati trasformati, ma il 'vectorizer' stesso.
# Perché? Se domani arriva un nuovo documento, dobbiamo trasformarlo usando gli STESSI 
# parametri e lo STESSO vocabolario imparato oggi per mantenere la coerenza dimensionale.
model_filename = "tfidf_processor_v2026.pkl"
joblib.dump(vectorizer, model_filename)

print(f"\n[INFO] Pipeline salvata in: {model_filename}")
print("[INFO] Il file può ora essere caricato in una web-app per processare nuovi input utente.")