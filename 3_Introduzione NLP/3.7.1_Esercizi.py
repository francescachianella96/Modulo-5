import os
import nltk
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import Counter

# Download delle risorse NLTK necessarie per la tokenizzazione
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def analyze_overstemming():
    """
    Script per analizzare l'impatto dello stemming sulla semantica.
    Teoria: Lo stemming è un processo euristico che taglia le desinenze.
    L'over-stemming avviene quando termini con significati diversi vengono ricondotti alla stessa radice.
    """
    
    # Frase fornita dalla traccia
    frase = "Il fioraio ha venduto i fiori e ha osservato la fioritura nel giardino."
    
    # 1. TOKENIZZAZIONE
    # Trasformiamo la frase in una lista di parole (tokens)
    tokens = word_tokenize(frase.lower(), language='italian')
    # Rimuoviamo la punteggiatura per un'analisi pulita
    tokens = [w for w in tokens if w.isalpha()]
    
    print(f"Testo originale: {frase}")
    print(f"Token estratti: {tokens}\n")

    # 2. APPLICAZIONE SNOWBALL STEMMER (ITALIANO)
    stemmer = SnowballStemmer(language='italian')
    
    stems = []
    mapping = {} # Dizionario per visualizzare Parola -> Stem
    
    for word in tokens:
        root = stemmer.stem(word)
        stems.append(root)
        mapping[word] = root

    # 3. CONTEGGIO RADICI UNICHE
    unique_stems = set(stems)
    
    print("--- RISULTATI DELLO STEMMING ---")
    for word, root in mapping.items():
        print(f"{word:12} -> {root}")
    
    print(f"\nNumero di parole originali: {len(tokens)}")
    print(f"Numero di radici (stems) uniche: {len(unique_stems)}")

    # 4. IDENTIFICAZIONE COLLISIONI (Fiori vs Fioritura)
    stem_fiori = stemmer.stem("fiori")
    stem_fioritura = stemmer.stem("fioritura")
    
    print("\n--- ANALISI SPECIFICA ---")
    print(f"Stem di 'fiori':     {stem_fiori}")
    print(f"Stem di 'fioritura': {stem_fioritura}")

    # 5. RIFLESSIONE TEORICA
    print("\n--- RIFLESSIONE SULL'OVER-STEMMING ---")
    if stem_fiori == stem_fioritura:
        print("RISCONTRO: Le parole sono collassate sulla stessa radice ('fior').")
        print("ANALISI: In questo caso siamo di fronte a un potenziale Over-stemming.")
        print("PRO: Aumenta il 'Recall' (se cerco 'fiori' trovo documenti sulla 'fioritura').")
        print("CONTRO: Si perde la distinzione tra l'OGGETTO (fiori) e il PROCESSO (fioritura).")
        print("CONCLUSIONE: Per un sistema di Information Retrieval generico è un bene, "
              "ma per un sistema di analisi biologica o tecnica è un male perché "
              "elimina la precisione temporale e ontologica del termine.")

if __name__ == "__main__":
    analyze_overstemming()