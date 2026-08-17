from transformers import BertTokenizer, BertModel
import torch
import tensorflow as tf

# 1. Caricamento del modello e del tokenizer 'bert-base-uncased'
# Utilizziamo la libreria transformers di Hugging Face
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 2. Preparazione di una frase di esempio e conversione in tensori
frase = "I love this course about Deep Learning!"
# 'return_tensors' specifica il framework di output (pt = PyTorch)
inputs = tokenizer(frase, return_tensors="pt")

# Stampiamo gli input per vedere come sono strutturati (input_ids, attention_mask)
print(f"Tokenized Inputs:\n{inputs}\n")

# 3. Estrazione dell'ultimo stato nascosto (last_hidden_state)
# Passiamo gli input al modello senza calcolare i gradienti (per risparmiare memoria)
with torch.no_grad():
    outputs = model(**inputs)

# last_hidden_state ha dimensione [batch_size, sequence_length, hidden_size]
last_hidden_state = outputs.last_hidden_state
print(f"Shape di last_hidden_state: {last_hidden_state.shape}")

# 4. Isolare il vettore corrispondente al token '[CLS]'
# Il token '[CLS]' è sempre il primo token della sequenza (indice 0)
# La forma sarà [batch_size, hidden_size]
cls_vector = last_hidden_state[:, 0, :]
print(f"Shape del vettore [CLS]: {cls_vector.shape}\n")

# 5. Ipotizzare la creazione di un layer denso in Keras
# Supponiamo di voler fare una classificazione binaria (Sentiment Analysis)
# Il vettore [CLS] funge da rappresentazione aggregata dell'intera frase
input_dim = cls_vector.shape[1]  # Solitamente 768 per BERT base

dense_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', name='sentiment_classifier')

# Esempio di applicazione (ipotetico)
# In una pipeline reale, convertiremmo il tensore PyTorch in un array NumPy o tensore TF
cls_vector_tf = tf.convert_to_tensor(cls_vector.numpy())
prediction = dense_layer(cls_vector_tf)

print("--- Mock Keras Classification ---")
print(f"Layer Dense creato: {dense_layer}")
print(f"Output della classificazione (probabilità): {prediction.numpy()[0][0]:.4f}")