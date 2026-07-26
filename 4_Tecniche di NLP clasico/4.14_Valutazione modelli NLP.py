"""
================================================================================
ESEMPIO PRATICO: VALUTAZIONE DEI MODELLI DI NLP - OLTRE L'ACCURATEZZA
================================================================================
Questo script è stato progettato per dimostrare come valutare un classificatore
di testi (Filtro Spam) utilizzando Keras 3 con backend PyTorch (Best Practice 2026).

Vengono approfonditi i concetti delle slide:
- Matrice di Confusione (TP, FP, FN, TN)
- Precision, Recall e F1-Score
- Trade-off della Soglia (Tau)
- Curve ROC-AUC e Precision-Recall
- Analisi qualitativa degli errori (Falsi Positivi)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Importiamo le metriche di scikit-learn per la diagnostica professionale
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    fbeta_score, 
    roc_curve, 
    auc, 
    precision_recall_curve
)

# ------------------------------------------------------------------------------
# 1. SETUP DEL BACKEND (Best Practice 2026)
# ------------------------------------------------------------------------------
# Keras 3 è agnostico rispetto al framework. Impostiamo PyTorch come motore di calcolo.
# Questa riga DEVE essere eseguita prima di importare 'keras'.
os.environ["KERAS_BACKEND"] = "torch"
import keras

# ------------------------------------------------------------------------------
# 2. GESTIONE DATI (Preprocessing & Caricamento)
# ------------------------------------------------------------------------------

def carica_e_prepara_dati():
    """
    Scarica il dataset reale 'SMS Spam Collection' e lo prepara per PyTorch.
    
    Il dataset è composto da:
    - Label: 'ham' (posta lecita) o 'spam'
    - Message: Il testo del messaggio SMS
    """
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    # Lettura via Pandas (formato tab-separated)
    df = pd.read_csv(url, sep='\t', names=['label', 'message'])
    
    # Conversione testo: Forziamo il tipo stringa per evitare errori 'object' in PyTorch
    X = df['message'].astype(str).values
    
    # Conversione etichette:
    # PyTorch richiede float32 per calcolare la perdita (Loss) binary_crossentropy
    # 0 = Ham (Classe Negativa), 1 = Spam (Classe Positiva)
    y = df['label'].map({'ham': 0, 'spam': 1}).values.astype("float32")
    
    # Suddivisione Train/Test (80% / 20%)
    # 'stratify=y' garantisce che la proporzione di Spam sia uguale in entrambi i set
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------------------
# 3. ARCHITETTURA DEL MODELLO (Keras Functional API)
# ------------------------------------------------------------------------------

def build_model_nlp(max_tokens=5000, seq_len=50):
    """
    Crea un classificatore NLP neurale.
    Input: Sequenze di numeri interi (ID delle parole).
    """
    # L'input riceve vettori di interi (lunghezza fissa 50 parole)
    inputs = keras.Input(shape=(seq_len,), dtype="int32", name="input_sequenza")
    
    # Layer 1: Embedding - Trasforma ogni ID parola in un vettore denso di 16 dimensioni
    x = keras.layers.Embedding(input_dim=max_tokens, output_dim=16)(inputs)
    
    # Layer 2: Pooling - Riassume il significato del messaggio mediando i vettori delle parole
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Layer 3: Dense - Strato 'ragionante' con attivazione ReLU
    x = keras.layers.Dense(16, activation="relu")(x)
    
    # Layer 4: Dropout - Spegne casualmente neuroni per evitare overfitting (memorizzazione)
    x = keras.layers.Dropout(0.1)(x)
    
    # Layer 5: Output - Sigmoid restituisce una probabilità tra 0 e 1 (Slide 10)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="probabilita")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compilazione: usiamo Adam (ottimizzatore standard) e Binary Crossentropy
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ------------------------------------------------------------------------------
# 4. FUNZIONI DI DIAGNOSTICA (Visualizzazione & Statistiche)
# ------------------------------------------------------------------------------

def mostra_diagnostica_avanzata(y_true, y_probs):
    """
    Visualizza le Curve ROC e Precision-Recall.
    Questi grafici mostrano come il modello performa PER TUTTE le possibili soglie.
    """
    # Calcolo coordinate Curva ROC e area AUC (capacità di separazione classi)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    # Calcolo Curva Precision-Recall (quanto è 'pulito' il recupero dei rari spam)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Grafico ROC: Un modello perfetto va subito in alto a sinistra (AUC=1.0)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--') # Linea del caso (50/50)
    ax1.set_title('Capacità Discrimante (ROC)')
    ax1.set_xlabel('False Positive Rate (Falsi Allarmi)')
    ax1.set_ylabel('True Positive Rate (Messaggi Intercettati)')
    ax1.legend(loc="lower right")

    # Grafico P-R: Mostra la sfida tra purezza e completezza
    ax2.plot(recall, precision, color='blue', lw=2, label='P-R curve')
    ax2.set_title('Purezza vs Completezza (P-R)')
    ax2.set_xlabel('Recall (Quanti Spam prendo?)')
    ax2.set_ylabel('Precision (Quanto sono affidabile?)')
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
    return roc_auc

def mostra_diagnostica_soglia(y_true, y_probs, soglia=0.5):
    """
    Analizza il modello scegliendo una specifica soglia decisionale (Tau).
    Esegue il passaggio fondamentale Probabilità -> Classe Finale.
    """
    # Trasformazione: Se prob > soglia allora è 1 (Spam), altrimenti 0
    y_pred = (y_probs > soglia).astype(int)
    
    print(f"\n" + "="*55)
    print(f" ANALISI OPERATIVA - SOGLIA DECISIONALE (Tau) = {soglia}")
    print("="*55)
    
    # Report testuale con Precision, Recall e F1-Score per ogni classe
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    
    # Matrice di Confusione (Slide 4): Rappresentazione visuale dei 4 quadranti
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='coolwarm', cbar=False)
    plt.title(f'Matrice di Confusione (Soglia {soglia})')
    plt.ylabel('Realtà')
    plt.xlabel('Previsione')
    plt.show()

def analizza_falsi_positivi(X_raw, y_true, y_probs, soglia=0.5):
    """
    IDENTIFICAZIONE ERRORI: Trova i messaggi reali scambiati per spam.
    È l'analisi più importante per non distruggere l'esperienza utente.
    """
    y_pred = (y_probs > soglia).astype(int).flatten()
    
    # Logica booleana: realtà=0 (Lecito) AND previsione=1 (Spam)
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    
    print(f"\n" + "!"*55)
    print(f" FOCUS ERRORI: I 'FALSI ALLARMI' ALLA SOGLIA {soglia}")
    print(f" Numero totale di messaggi leciti persi: {len(fp_indices)}")
    print("!"*55)
    
    if len(fp_indices) > 0:
        for i in fp_indices[:10]: # Limite a 10 per non intasare la console
            print(f"- [ERRORE]: {X_raw[i][:110]}...") 
    else:
        print("Ottimo! Nessun messaggio lecito è stato erroneamente bloccato.")

# ------------------------------------------------------------------------------
# 5. ESECUZIONE DEL WORKFLOW PROGRAMMATICO
# ------------------------------------------------------------------------------

# STEP 1: Acquisizione Dati
X_train_raw, X_test_raw, y_train, y_test = carica_e_prepara_dati()

# STEP 2: Vettorizzazione (Preprocessing)
# Creiamo un vocabolario delle 5000 parole più frequenti.
vectorizer = keras.layers.TextVectorization(
    max_tokens=5000, 
    output_mode="int", 
    output_sequence_length=50 # Ogni SMS diventa un vettore di 50 numeri
)
# Analizziamo il testo del training per 'imparare' il vocabolario
vectorizer.adapt(X_train_raw)

# Trasformiamo le stringhe in matrici numeriche per il modello PyTorch
X_train = vectorizer(X_train_raw)
X_test = vectorizer(X_test_raw)

# STEP 3: Creazione e Addestramento
modello = build_model_nlp()
print("\n[AI] Addestramento del classificatore in corso...")
modello.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)

# STEP 4: Inferenza (Predizione)
# Il modello ci dà una lista di probabilità (es. 0.98, 0.12, 0.45...)
y_probs = modello.predict(X_test)

# STEP 5: Valutazione Globale (Curve e AUC)
auc_val = mostra_diagnostica_avanzata(y_test, y_probs)
print(f"\n[RISULTATO] Area Sotto la Curva (AUC): {auc_val:.4f}")

# STEP 6: Confronto Strategico delle Soglie (Slide 9-10)

# CASO A: Soglia Standard (Bilanciata)
# Qui il modello è più "coraggioso", intercetta più spam ma rischia più falsi allarmi.
SOGLIA_STANDARD = 0.5
mostra_diagnostica_soglia(y_test, y_probs, soglia=SOGLIA_STANDARD)
analizza_falsi_positivi(X_test_raw, y_test, y_probs, soglia=SOGLIA_STANDARD)

# CASO B: Soglia Prudente (Alta Precision)
# Alziamo la "manopola" per essere più sicuri prima di bloccare un messaggio.
# Ideale se vogliamo evitare a tutti i costi di censurare mail lecite.
SOGLIA_PRUDENTE = 0.85 
mostra_diagnostica_soglia(y_test, y_probs, soglia=SOGLIA_PRUDENTE)
analizza_falsi_positivi(X_test_raw, y_test, y_probs, soglia=SOGLIA_PRUDENTE)

# STEP 7: Verdetto Finale (F1-Score)

# Calcolo dell'ago della bilancia (Slide 11-12) alla soglia standard.
f1 = fbeta_score(y_test, (y_probs > 0.5).astype(int), beta=1)
print(f"\n[VERDETTO] F1-Score Finale (Equilibrio Precision/Recall): {f1:.4f}")