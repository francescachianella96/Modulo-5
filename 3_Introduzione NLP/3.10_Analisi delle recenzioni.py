import os
import re
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


# Caricamento del modello linguistico spaCy per l'italiano
# Teoria: 'it_core_news_sm' è un modello statistico multi-task che esegue
# simultaneamente tokenizzazione, POS tagging e lemmatizzazione.
try:
    nlp = spacy.load("it_core_news_sm")
except OSError:
    os.system("python -m spacy download it_core_news_sm")
    nlp = spacy.load("it_core_news_sm")

def full_nlp_pipeline(text_data):
    """
    Pipeline Integrata: Pulizia -> Tokenizzazione -> Lemmatizzazione.
    
    Teoria: La pipeline trasforma il segnale testuale da 'unstructured' a 'structured'
    riducendo l'entropia e la varianza morfologica del dataset.
    """
    processed_docs = []
    
    for text in text_data:
        # 1. Pulizia con Regex (Noise Removal)
        # Rimuoviamo URL e tag HTML prima che entrino nel modello linguistico.
        text = re.sub(r'<.*?>|https?://\S+', '', text)
        # Rimuoviamo caratteri speciali mantenendo solo lettere e spazi.
        text = re.sub(r'[^a-zA-Zàèìòù\s]', '', text).lower()
        
        # 2. Processing con spaCy (Tokenization + Lemmatization)
        # Il metodo nlp() trasforma la stringa in un oggetto Doc ricco di annotazioni.
        doc = nlp(text)
        
        # 3. Estrazione Lemmi e Rimozione Stopwords
        # Teoria: Filtriamo le parole ad alta frequenza ma basso valore semantico.
        # Estraiamo il .lemma_ per ricondurre ogni parola alla sua forma canonica.
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
        processed_docs.append(" ".join(lemmas))
        
    return processed_docs

# --- ESEMPIO DI DATASET: RECENSIONI HOTEL ---
raw_reviews = [
    "Il soggiorno è stato fantastico! <br> Personale gentilissimo e camere pulite.",
    "Esperienza pessima. La camera era sporca e il personale maleducato. Sconsiglio.",
    "Ottima posizione, ma il check-in è stato lentissimo. Camere nella media.",
    "Colazione incredibile! Tornerò sicuramente in questo hotel fantastico.",
    "Non mi è piaciuto nulla. Sporco ovunque e troppo rumore di notte."
]

# Esecuzione della Pipeline
print("Esecuzione Pipeline in corso...")
cleaned_reviews = full_nlp_pipeline(raw_reviews)

# --- ANALISI DELLE FREQUENZE ---
# Teoria: La Legge di Zipf suggerisce che pochi termini domineranno il corpus.
# L'analisi delle frequenze ci permette di validare la qualità della pulizia.
all_words = " ".join(cleaned_reviews).split()
word_freq = Counter(all_words)
common_words = word_freq.most_common(10)

print(f"\nTop 10 Lemmi più frequenti: {common_words}")

# --- VISUALIZZAZIONE: WORDCLOUD ---
# Teoria: La WordCloud è una mappatura spaziale dove l'area occupata dal testo
# è proporzionale alla sua frequenza relativa nel dataset.
wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='white',
    colormap='viridis'
).generate(" ".join(cleaned_reviews))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Visualizzazione Semantica delle Recensioni")
plt.show()
