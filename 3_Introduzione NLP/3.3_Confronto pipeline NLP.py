import os
import nltk
import spacy
from transformers import pipeline


# 1. APPROCCIO NLTK (Accademico e Granulare)
# Teoria: NLTK è una libreria 'symbolic-first'. Ogni operazione è un modulo isolato.
def run_nltk_demo(text):
    # Scarichiamo i tokenizer necessari. 
    # 'punkt_tab' è richiesto dalle versioni più recenti di NLTK per la tokenizzazione.
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) # FIX: Aggiunto per risolvere l'errore Resource Not Found
    
    # Tokenizzazione: divide il testo in unità minime (token) usando regole grammaticali
    tokens = nltk.word_tokenize(text)
    print(f"[NLTK Tokens]: {tokens}")

# 2. APPROCCIO SPACY (Industriale 'Production-Ready')
# Teoria: spaCy crea un oggetto 'Doc' che contiene tutte le annotazioni linguistiche
# calcolate in un unico passaggio tramite una pipeline ottimizzata in Cython.
def run_spacy_demo(text):
    # Carichiamo il modello per l'italiano (small)
    # BEST PRACTICE: Usare modelli pre-installati per evitare latenza al primo avvio
    try:
        nlp = spacy.load("it_core_news_sm")
    except OSError:
        # Fallback nel caso il modello non sia presente nell'ambiente
        print("Modello spaCy non trovato. Download in corso...")
        os.system("python -m spacy download it_core_news_sm")
        nlp = spacy.load("it_core_news_sm")
    
    doc = nlp(text)
    
    # NER (Named Entity Recognition): Identifica entità come Persone, Luoghi o Organizzazioni
    # Teoria: spaCy usa modelli statistici per inferire il ruolo dei nomi nel contesto
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"[spaCy NER]: {entities}")

# 3. APPROCCIO HUGGING FACE (SOTA Deep Learning)
# Teoria: Hugging Face astrae i modelli Transformer (BERT, GPT, ecc.)
# L'oggetto 'pipeline' gestisce automaticamente tokenizzazione e inferenza neurale.
def run_transformers_demo(text):
    # Sentiment Analysis: determina l'emozione prevalente nel testo
    # Teoria: I modelli Transformer usano l'Attention Mechanism per pesare l'importanza delle parole
    print("Inizializzazione Transformer Pipeline...")
    
    # FIX: Aggiunto framework="pt" per forzare PyTorch ed evitare il conflitto con Keras 3
    classifier = pipeline("sentiment-analysis", model="dbmdz/bert-base-italian-xxl-cased", framework="pt")
    
    result = classifier(text)
    print(f"[Hugging Face Sentiment]: {result}")


# ESECUZIONE
if __name__ == "__main__":
    test_text = "Leonardo da Vinci ha dipinto la Gioconda a Firenze, è un'opera meravigliosa."
    print("--- INIZIO ELABORAZIONE NLP ---")
    run_nltk_demo(test_text)
    print("-" * 30)
    run_spacy_demo(test_text)
    print("-" * 30)
    run_transformers_demo(test_text)