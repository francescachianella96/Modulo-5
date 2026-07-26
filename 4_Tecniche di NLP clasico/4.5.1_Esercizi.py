import gensim.downloader as api

# 2. CARICAMENTO MODELLO PRE-ADDESTRATO
# Teoria: Usiamo il modulo 'downloader' di Gensim per accedere a modelli standard.
# Il modello 'glove-twitter-25' è leggero e perfetto per scopi didattici.
# Nota: 'Glove' e 'Word2Vec' sono tecnicamente diversi ma condividono la logica di lookup vettoriale.
print("Download e caricamento dello spazio vettoriale in corso...")
word_vectors = api.load("glove-wiki-gigaword-100") 

# 1. Similitudine semplice
print("Simili a 'computer':", word_vectors.most_similar("computer", topn=3))

# 2. Analogia geografica
# Roma - X + Spagna = Madrid  => X = Roma + Spagna - Madrid
# In Gensim: positive=[Roma, Spagna], negative=[Madrid]
geo_result = word_vectors.most_similar(positive=['rome', 'spain'], negative=['madrid'], topn=1)
print(f"L'analogia suggerisce: {geo_result[0][0]}")

# 3. Identificazione intruso
# Teoria: L'intruso è il vettore che ha la distanza media maggiore rispetto al baricentro del gruppo.
intrusa = word_vectors.doesnt_match(["apple", "pear", "banana", "car"])
print(f"L'intrusa nel gruppo è: {intrusa}")

# Spiegazione del perché: 
# apple, pear e banana formano un cluster semantico compatto (frutta). 
# 'car' si trova in una regione dello spazio vettoriale dedicata ai trasporti, 
# matematicamente molto distante dal cluster dei frutti.