from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "il paziente mostra sintomi di influenza e febbre alta",
    "influenza stagionale si combatte con il riposo"
]

v = TfidfVectorizer()
res = v.fit_transform(docs)

# Estrazione dati per il primo documento (indice 0)
feature_names = v.get_feature_names_out()
scores = res.toarray()[0]
ranking = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

print("Classifica TF-IDF per il Documento 1:")
for word, score in ranking[:5]:
    print(f"{word}: {score:.4f}")

"""
SPIEGAZIONE TEORICA AGGIUNTIVA:
La parola 'influenza' appare in entrambi i documenti, quindi la sua Inverse Document Frequency (IDF) 
sarà più bassa (è meno 'rara'). La parola 'paziente' appare solo nel primo documento: 
questo le conferisce un'IDF più alta. 

Anche se entrambe hanno la stessa TF (appaiono 1 volta), 'paziente' vince perché è più 
DISTINTIVA del primo documento rispetto all'intero corpus.
"""