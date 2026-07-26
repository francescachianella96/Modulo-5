from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# 1. Dataset
data = [
    "L'AI è il futuro del lavoro",
    "Il Deep Learning è una branca dell'AI",
    "I modelli di AI sono molto potenti",
    "Il lavoro del futuro sarà basato sull'AI",
    "Deep Learning e computer vision"
]

# 2. Configurazione: min_df=2 ignora termini troppo rari (rumore)
# max_features=100 controlla l'occupazione di memoria
vec = TfidfVectorizer(min_df=2, max_features=100)

# 3. Fit e Salvataggio
vec.fit(data)
joblib.dump(vec, "exercise_vec.joblib")

# 4. Inferenza
loaded = joblib.load("exercise_vec.joblib")
frase = ["Il Deep Learning è il cuore dell'AI"]
vettore = loaded.transform(frase)

# Analisi risultati
print(f"Feature totali nel vocabolario: {len(loaded.vocabulary_)}")
print(f"Indici delle feature attive nel vettore: {vettore.indices}")
print(f"Pesi TF-IDF corrispondenti: {vettore.data}")

# Nota teorica: Termini come 'cuore' verranno ignorati perché non presenti nel 
# vocabolario costruito durante il fit (Out-of-Vocabulary).