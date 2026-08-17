"""
================================================================================
SEQUENCE-TO-SEQUENCE (Seq2Seq) CON DATASET REALE
================================================================================
Obiettivo: Tradurre frasi dall'Inglese all'Italiano (EN -> IT).
Architettura: Encoder-Decoder con LSTM Bidirezionale .
Dataset: Hugging Face 'opus_books'.

"""

import os

# --- CONFIGURAZIONE AMBIENTE ---
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import layers
import torch

# Verifica diagnostica
print(f"--- DIAGNOSTICA AMBIENTE ---")
print(f"Backend Keras: {keras.backend.backend()}")
print(f"GPU Disponibile (PyTorch): {torch.cuda.is_available()}")

def scarica_e_prepara_dati(num_samples=5000):
    """
    Gestisce il reperimento dei dati e la loro trasformazione in numeri (vettorizzazione).
    """
    try:
        from datasets import load_dataset
        print(f"--- 1. CARICAMENTO DATI ---")
        print(f"Recupero {num_samples} frasi da Hugging Face...")
        
        # Scarichiamo il dataset specializzato in traduzioni LIBRI (coppia en-it)
        dataset = load_dataset("opus_books", "en-it", split=f"train[:{num_samples}]", trust_remote_code=True)
        
        # Estraggono le frasi e le normalizziamo in minuscolo
        input_texts = [ex["translation"]["en"].lower() for ex in dataset]
        
        # Per il Decoder, aggiungiamo dei "marcatori di controllo":
        # 'starttoken' -> Dice al decoder: "Comincia a tradurre ora!"
        # 'endtoken'   -> Dice al decoder: "Hai finito, fermati!"
        target_texts = [f"starttoken {ex['translation']['it'].lower()} endtoken" for ex in dataset]
        print(f"Caricamento completato: {len(input_texts)} esempi pronti.")
        
    except Exception as e:
        print(f"\n[ERRORE] Hugging Face non disponibile: {e}. Uso dati sintetici.")
        input_texts = ["i am happy", "it is hot"] * (num_samples // 2)
        target_texts = ["starttoken sono felice endtoken", "starttoken fa caldo endtoken"] * (num_samples // 2)

    # --- 2. VETTORIZZAZIONE (Trasformazione testo -> numeri) ---
    max_tokens = 10000     # Dimensione massima del vocabolario (le 10k parole più comuni)
    sequence_length = 20   # Lunghezza fissa per ogni frase (se più corta, aggiunge zeri; se più lunga, taglia)

    # Vettorizzatore per l'Inglese (Input)
    src_vec = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int", # Converte ogni parola in un numero intero unico
        output_sequence_length=sequence_length,
    )
    
    # Vettorizzatore per l'Italiano (Output)
    tgt_vec = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length + 1, # Un passo in più per gestire lo shift temporale
    )

    # 'adapt' legge i testi per creare il dizionario parole-numeri
    src_vec.adapt(input_texts)
    tgt_vec.adapt(target_texts)

    return src_vec, tgt_vec, input_texts, target_texts

def costruisci_modello_seq2seq(src_vec, tgt_vec, latent_dim=512):
    """
    COSTRUZIONE DELL'ARCHITETTURA ENCODER-DECODER (Seq2Seq).
    
    Questo modello è composto da due parti principali che lavorano in tandem:
    1. ENCODER: Legge la frase in inglese e la 'condensa' in un vettore di stato (pensiero).
    2. DECODER: Prende quel vettore e 'srotola' la traduzione parola per parola.
    """
    
    # Recuperiamo il numero di parole uniche (token) dai vettorizzatori.
    # Serve per dimensionare correttamente gli strati di ingresso (Embedding).
    num_src_tokens = src_vec.vocabulary_size()
    num_tgt_tokens = tgt_vec.vocabulary_size()

    # ==========================================================================
    # PARTE A: L'ENCODER (La Comprensione)
    # --------------------------------------------------------------------------
    
    # 1. INPUT: Definiamo l'entrata per le sequenze di numeri interi (ID delle parole inglesi).
    # 'shape=(None,)' significa che accettiamo frasi di qualsiasi lunghezza.
    encoder_inputs = layers.Input(shape=(None,), dtype="int64", name="input_inglese")
    
    # 2. EMBEDDING: Trasforma ogni ID numerico in un vettore matematico denso di dimensione 'latent_dim'.
    # mask_zero=True è CRUCIALE: dice al modello di ignorare i PAD (zeri) usati per pareggiare le lunghezze.
    # Questo permette alla rete di concentrarsi solo sulle parole reali.
    x = layers.Embedding(num_src_tokens, latent_dim, mask_zero=True)(encoder_inputs)
    
    # 3. REGOLARIZZAZIONE: Spegniamo il 20% dei neuroni casualmente per evitare l'overfitting.
    x = layers.Dropout(0.2)(x)
    
    # 4. LSTM BIDIREZIONALE: È il 'cuore' dell'Encoder.
    # Legge la frase in avanti e all'indietro contemporaneamente.
    # return_state=True: Ci servono gli STATI FINALI (h e c) non solo gli output.
    # f_h, f_c: Stati del passaggio in avanti (Forward).
    # b_h, b_c: Stati del passaggio all'indietro (Backward).
    encoder_lstm = layers.Bidirectional(layers.LSTM(latent_dim, return_state=True))
    _, f_h, f_c, b_h, b_c = encoder_lstm(x)
    
    # 5. CONCATENAZIONE: Uniamo i due mondi (avanti e indietro).
    # Poiché abbiamo usato una LSTM Bidirezionale, dobbiamo unire gli stati.
    # Se latent_dim era 512, ora avremo un vettore di 1024 (512+512).
    state_h = layers.Concatenate()([f_h, b_h]) # Stato nascosto (memoria a breve termine)
    state_c = layers.Concatenate()([f_c, b_c]) # Stato della cella (memoria a lungo termine)
    
    # Questi due vettori insieme formano l'istante finale di comprensione della frase inglese.
    encoder_states = [state_h, state_c] 

    # ==========================================================================
    # PARTE B: IL DECODER (La Generazione)
    # --------------------------------------------------------------------------
    
    # 1. INPUT: L'ingresso per la traduzione italiana prodotta finora.
    # Durante il training usiamo il 'Teacher Forcing': diamo in pasto la risposta corretta shiftata.
    decoder_inputs = layers.Input(shape=(None,), dtype="int64", name="input_italiano_parziale")
    
    # 2. EMBEDDING ITALIANO: Proietta le parole italiane nello spazio vettoriale.
    x = layers.Embedding(num_tgt_tokens, latent_dim, mask_zero=True)(decoder_inputs)
    x = layers.Dropout(0.2)(x)
    
    # 3. LSTM DEL DECODER: Deve avere dimensione doppia (latent_dim * 2) perché
    # deve ospitare gli stati concatenati che arrivano dall'encoder bidirezionale.
    # return_sequences=True: Vogliamo l'output per OGNI parola della sequenza.
    decoder_lstm = layers.LSTM(latent_dim * 2, return_sequences=True, return_state=True)
    
    # 4. COLLEGAMENTO: Qui avviene la magia. Inizializziamo la memoria del decoder
    # con gli stati (h, c) prodotti dall'encoder. È il passaggio del testimone.
    decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
    
    # 5. STRATO DENSO FINALE: Trasforma i vettori interni (1024) in una distribuzione di
    # probabilità su tutto il vocabolario italiano (es. 10.000 parole).
    # activation="softmax": La somma di tutte le probabilità sarà 1.0.
    decoder_dense = layers.Dense(num_tgt_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # --------------------------------------------------------------------------
    # CREAZIONE DEI MODELLI FUNZIONALI (Training vs Inference)
    # --------------------------------------------------------------------------
    # Perché creiamo 3 modelli invece di uno solo?
    # - In Training, diamo la frase intera (molto veloce, calcolo parallelo).
    # - In Inferenzia (traduzione), dobbiamo andare una parola alla volta (più complesso).

    # 1. MODELLO DI TRAINING (End-to-End)
    # -----------------------------------
    # Questo è il modello "maestro". Viene usato solo durante .fit().
    # Prende due ingressi: la frase inglese e la frase italiana (Teacher Forcing).
    # L'output è la previsione di tutta la traduzione slittata di un passo.
    model = keras.Model(
        inputs=[encoder_inputs, decoder_inputs], # Due entrate: sorgente e target parziale
        outputs=decoder_outputs,                # Una uscita: la probabilità delle parole italiane
        name="modello_training"
    )
    
    # 2. MODELLO ENCODER (Estrattore di Significato)
    # ----------------------------------------------
    # In fase di produzione (traduzione di una nuova frase), ci serve isolare l'Encoder.
    # Questo modello prende 'encoder_inputs' (la frase inglese) e restituisce
    # 'encoder_states' (il famoso "Thought Vector" con gli stati H e C della LSTM).
    # Non ci serve l'output della LSTM qui, ma solo la sua "memoria finale".
    encoder_model = keras.Model(
        inputs=encoder_inputs, 
        outputs=encoder_states, 
        name="encoder_solo"
    )
    
    # 3. MODELLO DECODER (Generatore Iterativo)
    # -----------------------------------------
    # Questa è la parte più complessa. Nella traduzione reale, il decoder non riceve
    # gli stati dall'encoder una sola volta, ma deve aggiornare la propria memoria
    # a ogni parola generata. Quindi creiamo un modello che "vive un passo alla volta".

    # Definiamo due nuovi Input per ricevere gli stati (H e C) "dall'esterno" 
    # (ovvero dal loop di traduzione che vedremo dopo).
    # La dimensione è latent_dim * 2 perché l'encoder era Bidirezionale.
    dec_state_h = layers.Input(shape=(latent_dim * 2,), name="stato_h_manuale")
    dec_state_c = layers.Input(shape=(latent_dim * 2,), name="stato_c_manuale")
    dec_states_inputs = [dec_state_h, dec_state_c]
    
    # Riapplichiamo gli strati già creati sopra (Embedding, LSTM, Dense).
    # È FONDAMENTALE riutilizzare gli stessi oggetti layer (decoder_lstm, decoder_dense)
    # così che i pesi imparati nel training siano gli stessi usati qui.
    
    # Passiamo la parola corrente attraverso l'embedding del decoder.
    dec_x = layers.Embedding(num_tgt_tokens, latent_dim, mask_zero=True)(decoder_inputs)
    
    # Chiamiamo la LSTM del decoder, ma stavolta passiamo 'initial_state' manualmente
    # usando gli input che abbiamo appena definito (dec_states_inputs).
    # Riceviamo in output: la previsione (dec_out) e i NUOVI stati aggiornati (s_h, s_c).
    dec_out, s_h, s_c = decoder_lstm(dec_x, initial_state=dec_states_inputs)
    
    # Trasformiamo l'uscita della LSTM in probabilità di parole reali.
    dec_out = decoder_dense(dec_out)
    
    # Infine, assembliamo il modello decoder per l'inferenza:
    # INPUT: [Parola attuale] + [Stati H e C precedenti]
    # OUTPUT: [Previsione parola successiva] + [Nuovi stati H e C]
    decoder_model = keras.Model(
        inputs=[decoder_inputs] + dec_states_inputs, 
        outputs=[dec_out, s_h, s_c], 
        name="decoder_solo"
    )

    return model, encoder_model, decoder_model

def translate(sentence, encoder, decoder, src_vec, tgt_vec):
    """
    LOGICA DI INFERENZA (TRADUZIONE):
    Poiché il modello lavora con sequenze, non possiamo tradurre tutto in un colpo solo.
    Dobbiamo generare una parola, aggiungerla alla frase, e usarla per generare la successiva.
    """
    
    # 1. ENCODING: Passiamo la frase inglese nell'Encoder.
    # Otteniamo il 'Thought Vector' (gli stati h e c) che rappresentano il significato.
    states = encoder.predict(src_vec([sentence]), verbose=0)
    
    # 2. DIZIONARIO: Prepariamo lo strumento per trasformare i numeri in parole leggibili.
    target_vocab = tgt_vec.get_vocabulary()
    lookup = dict(zip(range(len(target_vocab)), target_vocab))
    
    # 3. INIZIALIZZAZIONE: La prima parola passata al decoder è sempre 'starttoken'.
    # Creiamo una sequenza vuota che contiene solo il token di inizio.
    token_seq = np.zeros((1, 1))
    token_seq[0, 0] = target_vocab.index("starttoken")
    
    decoded_sentence = ""
    
    # 4. LOOP DI GENERAZIONE: Generiamo una parola alla volta (max 20 parole).
    for _ in range(20):
        # Chiediamo al decoder: "Dato questo pensiero e l'ultima parola scritta, cosa viene dopo?"
        output_tokens, h, c = decoder.predict([token_seq] + [states[0], states[1]], verbose=0)
        
        # Scegliamo la parola con la probabilità più alta (Argmax).
        sampled_index = np.argmax(output_tokens[0, -1, :])
        word = lookup[sampled_index]
        
        # Se la parola generata è 'endtoken', il modello ha deciso che la frase è finita.
        if word == "endtoken":
            break
            
        # Aggiungiamo la parola alla frase finale.
        decoded_sentence += " " + word
        
        # AGGIORNAMENTO STATO: L'output di questa iterazione diventa l'input della prossima.
        # Passiamo la parola appena generata e i nuovi stati h e c (la "memoria" aggiornata).
        token_seq[0, 0] = sampled_index
        states = [h, c]
        
    return decoded_sentence.strip()

# ==============================================================================
# BLOCCO DI ESECUZIONE (MAIN)
# ==============================================================================
if __name__ == "__main__":
    # 1. PREPARAZIONE DATI: Scarichiamo 5000 coppie di frasi.
    src_v, tgt_v, raw_in, raw_tgt = scarica_e_prepara_dati(5000)
    
    # 2. CREAZIONE ARCHITETTURA: Istanziamo i 3 modelli (training, encoder, decoder).
    model, encoder_m, decoder_m = costruisci_modello_seq2seq(src_v, tgt_v)

    # 3. LOGICA 'TEACHER FORCING' PER IL TRAINING:
    # Il decoder durante il training non impara da solo parola per parola.
    # Gli diamo la frase italiana intera (enc_in) ma "slittata".
    
    # encoder_in: La frase inglese originale.
    enc_in = src_v(raw_in)
    
    # decoder_in: La frase italiana SENZA l'ultima parola (inizia con starttoken).
    # Serve come "guida" per il decoder.
    dec_in = tgt_v([t.rsplit(' ', 1)[0] for t in raw_tgt])
    
    # decoder_tgt: La frase italiana SENZA la prima parola (finisce con endtoken).
    # È quello che il decoder deve imparare a predire per ogni parola di dec_in.
    dec_tgt = tgt_v([t.split(' ', 1)[1] for t in raw_tgt])

    # 4. COMPILAZIONE: Usiamo Adam per l'ottimizzazione e Sparse Categorical Crossentropy.
    # 'Sparse' perché i nostri target sono numeri interi (ID parole) e non vettori One-Hot.
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # 5. TRAINING (IL MOMENTO DELLA VERITÀ):
    print("\n--- 3. ADDESTRAMENTO ---")
    print("Il modello imparerà a mappare il 'Pensiero' inglese sulla grammatica italiana.")
    # Passiamo [enc_in, dec_in] come input e dec_tgt come target.
    model.fit([enc_in, dec_in], dec_tgt, batch_size=64, epochs=50)

    # 6. TEST DI TRADUZIONE: Verifichiamo se ha imparato qualcosa.
    print("\n--- 4. TEST DI TRADUZIONE ---")
    test_phrases = ["i am happy", "the cat is black", "we love music"]
    for p in test_phrases:
        # Usiamo la funzione translate che usa i modelli di inference.
        traduzione = translate(p, encoder_m, decoder_m, src_v, tgt_v)
        print(f"INGLESE: '{p}'")
        print(f"ITALIANO: '{traduzione}'\n")