import os
import numpy as np
import gensim.downloader as api

# 1. CARICAMENTO DELLE STATISTICHE GLOBALI (GloVe)
# Teoria: GloVe (Global Vectors for Word Representation) è un algoritmo 
# di apprendimento non supervisionato per ottenere rappresentazioni vettoriali 
# delle parole. È stato addestrato su una matrice di co-occorrenza globale.
# Carichiamo una versione leggera (50 dimensioni) per scopi formativi.
print("Caricamento vettori GloVe da Gensim...")
glove_vectors = api.load("glove-wiki-gigaword-50")

from sklearn.metrics.pairwise import cosine_similarity

# 1. Recupero vettori
v_vapore = glove_vectors["steam"].reshape(1, -1)
v_caldo = glove_vectors["hoy"].reshape(1, -1)
v_freddo = glove_vectors["cold"].reshape(1, -1)

# 2. Calcolo similarità
sim_caldo = cosine_similarity(v_vapore, v_caldo)[0][0]
sim_freddo = cosine_similarity(v_vapore, v_freddo)[0][0]

print(f"Similarità vapore-caldo: {sim_caldo:.4f}")
print(f"Similarità vapore-freddo: {sim_freddo:.4f}")

# SPEIGAZIONE TEORICA AGGIUNTIVA:
# Matematicamente, vapore-caldo ha un punteggio maggiore perché nel corpus Wikipedia
# usato per GloVe, la co-occorrenza (apparizione ravvicinata) di "vapore" e "caldo" 
# è statisticamente più probabile rispetto a quella con "freddo". 
# GloVe cattura questo dato globale e lo codifica come vicinanza nello spazio vettoriale.