import os
import spacy
import numpy as np

# Configurazione Backend 
os.environ["KERAS_BACKEND"] = "torch"

# 1. CARICAMENTO MODELLO
try:
    nlp = spacy.load("it_core_news_md")
except OSError:
    print("Download modello in corso...")
    os.system("python -m spacy download it_core_news_md")
    nlp = spacy.load("it_core_news_md")

# 2. DEFINIZIONE DEI DOCUMENTI
# Contesto naturale
doc1 = nlp("Il gatto insegue il topo nel giardino.")
# Contesto "sporcato" da termini tecnici/alieni
doc2 = nlp("Il gatto insegue il topo nella stazione spaziale tramite un computer quantistico.")

# Definizione del termine di confronto (Target)
target = nlp("gatto")[0] # Estraiamo il token "gatto"

# 3. CALCOLO DELLA SIMILARITÀ
sim1 = target.similarity(doc1)
sim2 = target.similarity(doc2)

# 4. OUTPUT RISULTATI
print(f"Analisi di similarità per la parola: '{target.text}'\n")
print(f"1. Frase Standard: '{doc1.text}'")
print(f"   -> Similarità: {sim1:.4f}")

print(f"\n2. Frase Tecnica: '{doc2.text}'")
print(f"   -> Similarità: {sim2:.4f}")

# 5. DIMOSTRAZIONE DELLO SHIFT (Teoria del Centroide)
differenza = sim1 - sim2
print(f"\n--- Analisi ---")
print(f"Il 'rumore' semantico ha ridotto la similarità del {differenza*100:.2f}%.")