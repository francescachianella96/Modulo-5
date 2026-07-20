import keras
import nltk
from nltk.util import ngrams
from collections import Counter, defaultdict
import numpy as np

# Assicuriamoci di avere i dati per la tokenizzazione
nltk.download('punkt')

def build_ngram_statistical_model(corpus: str, n: int = 2):
    """
    Crea un modello statistico basato su N-grammi e calcola le probabilità condizionate.
    
    Teoria: Un modello di Markov di ordine N-1 assume che la probabilità di una parola 
    dipenda solo dagli N-1 token precedenti, riducendo la complessità computazionale.
    """
    
    # Tokenizzazione: trasformiamo il testo in una lista di atomi (unigrammi)
    tokens = nltk.word_tokenize(corpus.lower())
    
    # Generazione degli N-grammi
    # Teoria: usiamo una finestra scorrevole per catturare il contesto locale.
    # 
    generated_ngrams = list(ngrams(tokens, n))
    
    # Calcolo delle frequenze per il numeratore: Count(w_{n-1}, w_n)
    ngram_counts = Counter(generated_ngrams)
    
    # Calcolo delle frequenze per il denominatore (prefissi): Count(w_{n-1})
    # Se n=2, il prefisso è la singola parola precedente.
    prefix_counts = Counter([ng[:-1] for ng in generated_ngrams])
    
    # Calcolo delle probabilità condizionate: P(wn | wn-1) = Count(wn-1, wn) / Count(wn-1)
    # Utilizziamo un dizionario di dizionari per facilitare la predizione
    model = defaultdict(lambda: defaultdict(float))
    
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        target_word = ngram[-1]
        model[prefix][target_word] = count / prefix_counts[prefix]
        
    return model, generated_ngrams

# --- ESEMPIO PRATICO: ENTITÀ COMPOSTE ---
corpus_test = """
New York è una città incredibile. Molte persone visitano New York ogni anno. 
New York offre opportunità uniche, ma vivere a New York è costoso.
"""

# Analizziamo i Bigrammi (N=2)
model, bigrams_list = build_ngram_statistical_model(corpus_test, n=2)

print(f"Esempio di Bigrammi estratti: {bigrams_list[:5]}")

# Verifichiamo la probabilità che 'york' segua 'new'
prefix_key = ('new',)
prob_york = model[prefix_key]['york']

print(f"\nTeoria delle Probabilità:")
print(f"P('york' | 'new') = {prob_york:.2f}")
# Se la probabilità è 1.0, 'New' e 'York' formano un'entità bi-grammatica perfetta nel corpus.
