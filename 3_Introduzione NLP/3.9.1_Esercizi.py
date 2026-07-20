import nltk
from collections import Counter
from nltk.corpus import stopwords

# Scarichiamo le stopwords se non sono già presenti
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# testo
testo = "Il servizio non è buono, ma il cibo è molto buono."

# 1. Pulizia e Tokenizzazione
tokens_raw = nltk.word_tokenize(testo.lower())

# 2. Gestione Stopwords
# Definiamo le parole da rimuovere (es. 'è', 'il', 'ma')
# IMPORTANTE: In analisi del sentimento, spesso teniamo "non" e "molto" 
# perché cambiano il significato della frase.
stop_words = set(stopwords.words('italian'))
parole_da_mantenere = {'non', 'molto'}
stop_words_filtrate = [w for w in stop_words if w not in parole_da_mantenere]

# Filtriamo i token: rimuoviamo punteggiatura e stopwords
tokens = [t for t in tokens_raw if t.isalpha() and t not in stop_words_filtrate]

# 3. Generazione Bigrammi
bi_grams = list(nltk.bigrams(tokens))

# 4. Conteggi
counts = Counter(bi_grams)
prefix_counts = Counter(tokens)

# Ricerca delle occorrenze specifiche
count_non_buono = counts[('non', 'buono')]
count_molto_buono = counts[('molto', 'buono')]
total_non = prefix_counts['non']

# Calcolo Probabilità Condizionata: P(buono | non)
prob_cond = count_non_buono / total_non if total_non > 0 else 0

print(f"Token filtrati: {tokens}")
print(f"Bigrammi trovati: {bi_grams}")
print("-" * 30)
print(f"Occorrenze 'non buono': {count_non_buono}")
print(f"Occorrenze 'molto buono': {count_molto_buono}")
print(f"P(buono | non) = {prob_cond:.2f}")