import os

# Configurazione del Backend: Utilizziamo PyTorch come motore di calcolo per Keras 3.
# Questo permette di sfruttare le performance di Torch mantenendo la semplicità dell'API Keras.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================================================
# 1. SINUSOIDAL POSITIONAL ENCODING LAYER
# =========================================================================================
# I Transformer non hanno una struttura sequenziale intrinseca. La formula sinusoidale
# (Seno e Coseno) permette al modello di conoscere la posizione relativa e assoluta
# dei token senza dover "imparare" dei pesi aggiuntivi per le posizioni.
# =========================================================================================
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        # Token Embedding: Converte gli ID delle parole in vettori densi.
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Pre-calcoliamo la matrice delle posizioni usando la formula:
        # PE(pos, 2i) = sin(pos / 10000^(2i/d))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        self.positional_encoding = self._get_sinusoidal_encoding(sequence_length, embed_dim)

    def _get_sinusoidal_encoding(self, length, dim):
        # Generiamo le posizioni [0, 1, ..., length-1]
        pos = np.arange(length)[:, np.newaxis]
        # Generiamo gli indici delle dimensioni [0, 1, ..., dim-1]
        i = np.arange(dim)[np.newaxis, :]
        
        # Calcoliamo i "tassi d'angolo" (angle rates)
        # 2 * (i // 2) serve per avere lo stesso coefficiente sia per il seno che per il coseno della coppia
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim))
        angle_rads = pos * angle_rates

        # Applichiamo il seno agli indici pari e il coseno agli indici dispari
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        # Convertiamo in un tensore costante (non addestrabile)
        pos_encoding = angle_rads[np.newaxis, ...]
        return ops.cast(pos_encoding, dtype="float32")

    def call(self, inputs):
        # inputs shape: [Batch, Sequence_length]
        length = ops.shape(inputs)[-1]
        
        # Trasformiamo le parole in vettori
        embedded_tokens = self.token_embeddings(inputs)
        
        # Preleviamo la porzione di codifica posizionale necessaria
        # (se la frase è più corta della lunghezza massima)
        positions = self.positional_encoding[:, :length, :]
        
        # SOMMA: Uniamo l'informazione semantica (parola) con quella geometrica (seno/coseno).
        return embedded_tokens + positions

# =========================================================================================
# 2. COSTRUZIONE DEL MODELLO TRANSFORMER
# =========================================================================================
# Questa funzione definisce l'architettura. La particolarità è che restituisce due output:
# il sentiment (previsione) e la matrice di attenzione (per la spiegabilità).
# =========================================================================================
def build_transformer_model(vocab_size, maxlen, embed_dim, num_heads):
    # INPUT: Sequenza di interi (token ID) di lunghezza fissa 'maxlen'
    inputs = layers.Input(shape=(maxlen,), name="input_layer")
    
    # STEP 1: Embedding + Posizione
    # Trasformiamo gli ID in vettori che "sanno" dove si trovano nella frase.
    x = PositionalEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    
    # STEP 2: Multi-Head Attention (Il "Cuore" del Transformer)
    # Permette al modello di guardare diverse parti della frase simultaneamente.
    # 'return_attention_scores=True' è fondamentale per poter visualizzare la heatmap dopo.
    attention_output, attention_scores = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, name="attention_layer"
    )(x, x, return_attention_scores=True)
    
    # STEP 3: Estrazione della Matrice di Attenzione
    # Creiamo un "ponte" tramite un layer Activation lineare per marchiare l'output degli score.
    attention_matrix = layers.Activation("linear", name="attention_matrix")(attention_scores)
    
    # STEP 4: Riduzione e Classificazione
    # GlobalAveragePooling1D: Comprime la sequenza mediando i vettori delle parole.
    # Da [Sequence_length, Embed_dim] passiamo a un singolo vettore [Embed_dim].
    x = layers.GlobalAveragePooling1D()(attention_output)
    
    # LayerNormalization & Dropout: Regolarizzano prevenendo l'overfitting e stabilizzando i pesi.
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # OUTPUT FINALE: Sigmoide per classificazione binaria (0=Negativo, 1=Positivo).
    sentiment_output = layers.Dense(1, activation="sigmoid", name="sentiment_output")(x)

    # Il modello ha un ingresso e DUE uscite: previsione e pesi dell'attenzione.
    model = keras.Model(inputs=inputs, outputs=[sentiment_output, attention_matrix])
    return model

# =========================================================================================
# 3. COORDINATORE DELL'ESPERIMENTO
# =========================================================================================
def run_experiment():
    # Iperparametri
    vocab_size = 10000 # Prendiamo solo le 10.000 parole più comuni
    maxlen = 100      # Tronchiamo/Padderemo ogni recensione a 100 parole
    embed_dim = 64    # Dimensione dello spazio vettoriale delle parole
    num_heads = 4     # Numero di "prospettive" diverse nell'attenzione
    
    print("Pre-processing: Caricamento e padding del dataset IMDB...")
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    
    # Padding: Uniformiamo le frasi a lunghezza 100 inserendo zeri dove necessario.
    x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)
    
    # Istanziamo il modello tramite la funzione definita sopra
    model = build_transformer_model(vocab_size, maxlen, embed_dim, num_heads)
    
    # Compilazione: specifichiamo che la perdita (loss) va calcolata solo sul sentiment.
    # La matrice di attenzione non necessita di perdita (loss=None).
    model.compile(
        optimizer="adam",
        loss=["binary_crossentropy", None],
        metrics={"sentiment_output": "accuracy"}
    )

    print("\nInizio addestramento: Il modello impara la relazione parole-sentiment...")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=10)

    # Frasi di test scritte da noi per vedere se il modello ha "capito"
    custom_phrases = [
        "this movie was a masterpiece of cinema",
        "absolute waste of time and horrible acting",
    ]
    
    # Passiamo alla fase di visualizzazione
    visualize_attention(model, custom_phrases, maxlen)

# =========================================================================================
# 4. VISUALIZZAZIONE DEI PESI DELL'ATTENZIONE
# =========================================================================================
def visualize_attention(model, sentences, maxlen):
    # Scarichiamo il dizionario per convertire parole in ID come fatto nel dataset IMDB
    word_index = keras.datasets.imdb.get_word_index()
    
    for text in sentences:
        # 1. Tokenizzazione: Spezziamo la frase in parole e convertiamo in ID
        tokens = text.lower().split()
        # Nota: L'indice IMDB è shiftato di 3 (0=padding, 1=start, 2=oov)
        seq = [word_index.get(w, -3) + 3 for w in tokens]
        
        # 2. Padding a 100 per renderlo compatibile con l'input del modello
        padded = keras.utils.pad_sequences([seq], maxlen=maxlen)
        
        # 3. Predizione: Otteniamo sia il Sentiment che la Matrice di Attenzione
        pred, attn = model.predict(padded, verbose=0)
        
        # Estraiamo i pesi della PRIMA testa di attenzione (index 0)
        # La matrice ha forma [Batch, Heads, Seq_len, Seq_len]
        weights = ops.convert_to_numpy(attn)[0, 0] 
        
        # 4. Ritaglio: Poiché la frase è lunga 100 ma noi abbiamo scritto poche parole,
        # prendiamo solo l'angolo in basso a destra della matrice (le ultime 'n' parole).
        n = len(tokens)
        relevant_weights = weights[-n:, -n:]
        
        # 5. Plotting con Seaborn per creare la Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(relevant_weights, annot=True, fmt=".4f", cmap="YlGnBu",
                    xticklabels=tokens, yticklabels=tokens)
        
        # Mostriamo il sentiment predetto sopra il grafico
        sentiment_label = 'POSITIVO' if pred[0][0] > 0.5 else 'NEGATIVO'
        plt.title(f"Frase: '{text}'\nSentiment Predetto: {sentiment_label} (Confidenza: {pred[0][0]:.4f})")
        plt.xlabel("Parola su cui il modello si focalizza")
        plt.ylabel("Parola corrente analizzata")
        plt.tight_layout()
        plt.show()

# Punto di ingresso dello script
if __name__ == "__main__":
    run_experiment()