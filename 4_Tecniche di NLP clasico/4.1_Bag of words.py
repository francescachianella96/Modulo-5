import numpy as np

"""
PIPELINE DI ELABORAZIONE: BAG OF WORDS (BoW)
-------------------------------------------
Questo script implementa da zero il modello Bag of Words, una tecnica fondamentale
di elaborazione del linguaggio naturale (NLP) per convertire testo non strutturato
in dati numerici comprensibili dagli algoritmi di Machine Learning.

LOGICA DEL PROCESSO:
1. Tokenizzazione: Scomposizione del testo in singole parole (token).
2. Normalizzazione: Pulizia del testo (minuscolo, rimozione punteggiatura).
3. Vocabolario: Creazione di un elenco unico di parole note.
4. Vettorizzazione: Ogni documento diventa un vettore di frequenze basato sul vocabolario.
"""

# 1. DEFINIZIONE DEL CORPUS DI ESEMPIO
# Il 'corpus' è l'insieme di tutti i documenti (frasi) che vogliamo analizzare.
# Ogni stringa rappresenta un "documento" separato.
corpus = [
    "Il Deep Learning è affascinante.",
    "Le reti neurali leggono i dati.",
    "Il Deep Learning usa le reti neurali."
]

print("--- 1. Costruzione del Vocabolario ---")
# PASSO A: Pulizia e Tokenizzazione aggregata
# .join(corpus): Unisce tutte le frasi in un unico blocco di testo.
# .lower(): Uniforma il testo per evitare che "Il" e "il" siano contati come parole diverse.
# .replace(".", ""): Rimuove la punteggiatura che non porta significato semantico.
# .split(): Divide il testo in una lista di singole parole.
all_tokens = " ".join(corpus).lower().replace(".", "").split()

# PASSO B: Estrazione termini unici (Vocabolario)
# set(all_tokens): Elimina istantaneamente tutti i duplicati.
# sorted(list(...)): Ordina alfabeticamente per avere una mappatura coerente e prevedibile.
vocabulary = sorted(list(set(all_tokens)))

# PASSO C: Mapping Indice-Parola
# creiamo un dizionario dove ogni parola è associata a un ID numerico (la sua posizione nel vettore).
# Questo è fondamentale: la colonna 'n' della matrice corrisponderà sempre alla parola 'word_to_id[n]'.
word_to_id = {word: i for i, word in enumerate(vocabulary)}

print(f"Vocabolario ({len(vocabulary)} termini): {vocabulary}")
print(f"Mappatura: {word_to_id}\n")

# 2. CREAZIONE DELLA MATRICE DOCUMENT-TERM (DTM)
# Una DTM è una rappresentazione tabellare:
# - Righe (i): I singoli documenti del corpus.
# - Colonne (j): I termini presenti nel vocabolario.
# Inizializziamo una matrice di zeri con dimensioni [Num_Documenti x Dim_Vocabolario]
dtm = np.zeros((len(corpus), len(vocabulary)))

# POPOLAMENTO DELLA MATRICE (Vettorizzazione)
# Iteriamo su ogni documento per contare quante volte appare ogni parola del vocabolario.
for i, doc in enumerate(corpus):
    # Ripetiamo la pulizia per il singolo documento
    words = doc.lower().replace(".", "").split()
    for word in words:
        if word in word_to_id:
            # Identifichiamo la colonna corrispondente alla parola corrente
            index = word_to_id[word]
            # Incrementiamo il conteggio nella cella (riga documento, colonna parola)
            dtm[i, index] += 1

print("--- 2. Matrice Document-Term (BoW) ---")
# Ogni riga della matrice è ora la rappresentazione numerica della frase corrispondente.
print(dtm)

# 3. ANALISI DELLA SPARSITÀ
# Concetto Chiave: In NLP, se il vocabolario è di 50.000 parole e una frase ne contiene 10,
# il 99.9% della riga sarà composto da zeri. Questa è la "Sparsità".
# Formula: (Numero di zeri) / (Totale elementi della matrice)
num_nonzero = np.count_nonzero(dtm)
total_elements = dtm.size
sparsity = 1.0 - (num_nonzero / total_elements)

print(f"\n--- 3. Metriche del Dataset ---")
print(f"Elementi totali: {total_elements}")
print(f"Elementi non nulli: {num_nonzero}")
print(f"Sparsità della matrice: {sparsity * 100:.2f}%")
print("Nota: Un'alta sparsità indica che il modello BoW consuma molta memoria per pochi dati utili.")
