import keras
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from typing import List

# Download delle risorse NLTK necessarie
nltk.download('punkt')

def compare_stemmers(words: List[str], language: str = 'italian'):
    """
    Confronta l'efficacia di Porter e Snowball su una lista di termini.
    
    Teoria: Lo stemming è una 'regressione' morfologica. Riduce la varianza 
    aumentando però il rischio di bias (over-stemming).
    """
    
    # Inizializzazione degli stemmer
    # Porter: Il pioniere, basato su regole rigide per l'inglese.
    porter = PorterStemmer()
    
    # Snowball: Chiamato anche 'Porter2', più efficiente e multilingua.
    # Teoria: Snowball utilizza un linguaggio di programmazione specifico per 
    # descrivere algoritmi di stemming.
    snowball = SnowballStemmer(language=language)
    
    print(f"{'Parola Originale':<20} | {'Porter':<15} | {'Snowball (IT)':<15}")
    print("-" * 55)
    
    for word in words:
        p_stem = porter.stem(word)
        s_stem = snowball.stem(word)
        
        # Nota: Porter su parole italiane produrrà risultati spesso assurdi
        # perché cerca pattern morfologici inglesi (es: 'ing', 'ed', 's').
        print(f"{word:<20} | {p_stem:<15} | {s_stem:<15}")

# Dataset di test: Varianti verbali e potenziali casi di over-stemming
test_words = [
    "correre", "corriamo", "corressi",  # Varianti dello stesso verbo
    "università", "universo",           # Rischio over-stemming (radice comune?)
    "fiori", "fioraio", "fioritura",    # Famiglia semantica
    "andato", "andante"                 # Morfologia flessiva
]

print("--- ANALISI COMPARATIVA DELLO STEMMING ---")
compare_stemmers(test_words)
