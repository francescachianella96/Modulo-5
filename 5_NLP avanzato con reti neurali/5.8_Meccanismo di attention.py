"""
ESEMPIO 58: IL "FOCUS" DELL'INTELLIGENZA ARTIFICIALE (ATTENTION MECHANISM)
--------------------------------------------------------------------------
OBIETTIVO: Tradurre coppie "Colore + Oggetto" dall'Inglese (es. "red car") 
all'Italiano (es. "auto rossa"). 

COSA IMPAREREMO:
1. Come l'Encoder "legge" e crea una memoria.
2. Come il Decoder "scrive" consultando la memoria tramite l'Attention.
3. Come visualizzare il "lampo di genio" (i pesi di attenzione) che permette
   alla rete di capire che l'ordine delle parole va invertito.
"""

# --- PREPARAZIONE AMBIENTE ---
import os
# Spieghiamo a Keras di usare PyTorch come "motore" del calcolo (Best Practice 2026)
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, ops

# =================================================================
# 1. GENERAZIONE DATI: IL NOSTRO MINI-PROBLEMA DI TRADUZIONE
# =================================================================

def genera_dati_didattici(n_esempi=1000):
    """
    Crea un dataset dove l'ordine delle parole cambia tra Input e Output.
    Esempio: Input ["red", "car"] -> Output ["<START>", "auto", "rossa"]
    """
    # Vocabolari semplici
    colori = ['red', 'blue', 'green', 'yellow']
    oggetti = ['car', 'ball', 'bike', 'house']
    colori_it = ['rossa', 'blu', 'verde', 'gialla']
    oggetti_it = ['auto', 'palla', 'bici', 'casa']
    
    # Trasformiamo le parole in numeri (id) per la rete neurale
    voc_in = {p: i+1 for i, p in enumerate(colori + oggetti)}
    voc_out = {p: i+1 for i, p in enumerate(oggetti_it + colori_it)}
    voc_out['<START>'] = len(voc_out) + 1 # Segnale di inizio per il Decoder
    
    x, y = [], []
    for _ in range(n_esempi):
        c_idx, o_idx = np.random.randint(0, 4), np.random.randint(0, 4)
        # Input EN: [Colore, Oggetto]
        x.append([voc_in[colori[c_idx]], voc_in[oggetti[o_idx]]])
        # Output IT: [<START>, Oggetto, Colore] -> Notate l'inversione!
        y.append([voc_out['<START>'], voc_out[oggetti_it[o_idx]], voc_out[colori_it[c_idx]]])
        
    return np.array(x), np.array(y), voc_in, voc_out

# Carichiamo i dati e creiamo dizionari "inversi" per leggere i risultati
X, Y, voc_in, voc_out = genera_dati_didattici()
inv_voc_in = {i: p for p, i in voc_in.items()}
inv_voc_out = {i: p for p, i in voc_out.items()}

# =================================================================
# 2. ARCHITETTURA: IL DIALOGO TRA ENCODER, DECODER E ATTENTION
# =================================================================

def build_attention_model(input_vocab_size, output_vocab_size, embed_dim=32, latent_dim=64):
    """
    COSTRUZIONE DELLA RETE (PIANO DI ARCHITETTURA)
    Spiegazione delle dinamiche parola per parola:
    """
    
    # --- PARTE 1: L'ENCODER (La 'memoria a breve termine' che legge l'inglese) ---
    
    # Input: Porta d'ingresso. shape=(None,) significa: "accetta frasi di qualsiasi lunghezza".
    encoder_inputs = layers.Input(shape=(None,), name="Input_Inglese")
    
    # Embedding: Il 'Traduttore Concettuale'. Trasforma l'ID numerico della parola in un 
    # vettore di 32 numeri (embed_dim) che ne esprime il significato profondo.
    # mask_zero=True permette di ignorare eventuali padding (Best Practice per Seq2Seq).
    encoder_emb = layers.Embedding(input_vocab_size + 1, embed_dim, mask_zero=True)(encoder_inputs)
    
    # LSTM dell'Encoder: Il 'Lettore Attento'. 
    # - latent_dim: 64 'neuroni' che decidono cosa ricordare e cosa dimenticare.
    # - return_sequences=True: DINAMICA FONDAMENTALE. Dice alla LSTM: "Non darmi solo 
    #   il riassunto finale, ma lasciami un appunto per OGNI parola che leggi" (Slide 4).
    # - return_state=True: Chiede alla LSTM di consegnare anche i suoi 'stati interni' 
    #   finali (h e c), che serviranno per dare il 'contesto iniziale' al Decoder.
    encoder_outputs, state_h, state_c = layers.LSTM(
        latent_dim, return_sequences=True, return_state=True
    )(encoder_emb)
    
    # --- PARTE 2: IL DECODER (Lo 'scrittore' che genera l'italiano) ---
    
    # Input del Decoder: Riceve la parola italiana precedente per decidere la successiva.
    decoder_inputs = layers.Input(shape=(None,), name="Input_Italiano_Precedente")
    
    # Embedding del Decoder: Analogo a quello dell'encoder, ma per il vocabolario italiano.
    decoder_emb = layers.Embedding(output_vocab_size + 1, embed_dim, mask_zero=True)(decoder_inputs)
    
    # LSTM del Decoder: 
    # - initial_state=[state_h, state_c]: DINAMICA DI PASSAGGIO CONSEGNE. 
    #   Il Decoder inizia a pensare partendo esattamente da dove l'Encoder ha finito di leggere.
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=[state_h, state_c])
    
    # --- PARTE 3: MECCANISMO DI ATTENTION (Il 'Faro' o 'Torcia Elettrica') ---
    
    # layers.Attention: La classe che calcola la 'vicinanza' tra concetti (Slide 6).
    attention_layer = layers.Attention(name="Meccanismo_di_Attention")
    
    # Esecuzione Attention:
    # - [decoder_outputs, encoder_outputs]: DINAMICA DI CONFRONTO. 
    #   Mette in relazione "Cosa sto scrivendo ora" (Query) con "Tutto quello che è 
    #   stato letto" (Keys/Values).
    # - return_attention_scores=True: Ci restituisce i famosi 'Pesi Alpha' (Slide 10) 
    #   che useremo per disegnare la mappa colorata.
    context_vector, attention_weights = attention_layer(
        [decoder_outputs, encoder_outputs], 
        return_attention_scores=True
    )
    
    # --- PARTE 4: SINTESI E PREDIZIONE (Il verdetto finale) ---
    
    # Concatenate: DINAMICA DI UNIONE. Incolla insieme lo stato attuale del decoder 
    # con il 'Vettore di Contesto' (il distillato dei ricordi suggerito dall'Attention).
    decoder_combined_context = layers.Concatenate()([decoder_outputs, context_vector])
    
    # Dense con Softmax: DINAMICA DI SCELTA. Guarda il vettore unito e assegna una 
    # probabilità a ogni parola del vocabolario. Quella con il valore più alto 'vince'.
    output_dense = layers.Dense(output_vocab_size + 1, activation="softmax")
    predictions = output_dense(decoder_combined_context)
    
    # --- PARTE 5: CREAZIONE DEI MODELLI (I 'Cervelli' finali) ---
    
    # model: Il modello completo. Prende l'input inglese e quello italiano precedente 
    # per sfornare la predizione. Si usa per l'addestramento.
    model = keras.Model([encoder_inputs, decoder_inputs], predictions)
    
    # attention_visualizer: Modello 'Voyeur'. Invece della traduzione, ci sputa fuori 
    # solo i pesi dell'attenzione. Serve solo a noi umani per 'vedere' l'IA pensare.
    attention_visualizer = keras.Model([encoder_inputs, decoder_inputs], attention_weights)
    
    return model, attention_visualizer

# Creazione e Compilazione
full_model, att_model = build_attention_model(len(voc_in), len(voc_out))
full_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# =================================================================
# 3. ALLENAMENTO (TEACHER FORCING)
# =================================================================

print(">>> Fase 1: Addestramento della rete...")
# Y_in: Gli diamo la frase corretta troncata all'ultima parola (per imparare a prevedere la prossima)
# Y_out: La frase corretta che deve riuscire a generare
Y_in, Y_out = Y[:, :-1], Y[:, 1:]
full_model.fit([X, Y_in], Y_out, epochs=40, batch_size=32, verbose=0)
print(">>> Addestramento completato!")

# =================================================================
# 4. VISUALIZZAZIONE: APRIAMO LA SCATOLA NERA (Slide 11-12)
# =================================================================

def mostra_pensiero_ia(index=0):
    """
    Funzione per scattare una 'fotografia' a dove l'IA ha messo l'attenzione.
    """
    input_seq = X[index:index+1]
    target_seq = Y_in[index:index+1]
    
    # Chiediamo al modello di attenzione: "Fammi vedere i tuoi calcoli interni"
    # weights avrà forma (1, parole_out, parole_in)
    weights = att_model.predict([input_seq, target_seq], verbose=0)[0]
    
    # Prepariamo i testi per gli assi del grafico
    testo_in = [inv_voc_in[i] for i in input_seq[0]]
    testo_out = [inv_voc_out[i] for i in Y[index][1:]]
    
    # Disegniamo la mappa di calore (Heatmap)
    plt.figure(figsize=(7, 6))
    plt.imshow(weights, cmap='magma') # Mostriamo tutti i passi (niente più slice errati!)
    
    plt.xticks(range(len(testo_in)), testo_in, fontsize=12)
    plt.yticks(range(len(testo_out)), testo_out, fontsize=12)
    
    plt.title("MAPPA DI ATTENZIONE: Il Modello in Azione", pad=20)
    plt.xlabel("L'ENCODER legge l'Inglese (Input)", labelpad=10)
    plt.ylabel("Il DECODER scrive l'Italiano (Output)", labelpad=10)
    plt.colorbar(label="Intensità del Focus")
    plt.show()

# Eseguiamo la visualizzazione sul primo esempio
mostra_pensiero_ia(0)

# -------------------------------------------------------------------------
# GUIDA RAPIDA ALLA LETTURA DEL CODICE:
# -------------------------------------------------------------------------
# 1. PERCHÉ LSTM (return_sequences=True)?
#    Senza questo, l'Encoder darebbe solo l'ultimo stato (Slide 3: il collo di bottiglia). 
#    Con True, ogni parola ("red", "car") genera un vettore di memoria dedicato.
#
# 2. COS'È IL CONTEXT VECTOR? (Slide 14)
#    È il risultato del livello Attention. Se il modello sta scrivendo "auto", 
#    l'Attention peserà di più (es. 0.95) la memoria di "car" e meno (es. 0.05) quella 
#    di "red". Il Context Vector sarà "quasi uguale" al vettore di "car".
#
# 3. COME LEGGERE IL GRAFICO? (Slide 12)
#    Cerca i quadrati luminosi! Vedrai che quando l'IA scrive "auto" (asse Y), 
#    il quadrato più luminoso sarà in corrispondenza di "car" (asse X). 
#    Questo conferma che l'Attention funziona e ha imparato l'allineamento.