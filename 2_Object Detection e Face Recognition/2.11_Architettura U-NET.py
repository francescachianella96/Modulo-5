"""
================================================================================
IMPLEMENTAZIONE DI UNA MINI U-NET (KERAS 3 + PYTORCH BACKEND)
================================================================================
Questa implementazione dimostra l'architettura U-Net, uno standard nell'analisi 
di immagini biomediche e nella segmentazione semantica.

 ARCHITETTURA:
 1. Encoder (Contracting Path): Estrae feature semantiche riducendo la risoluzione.
 2. Bottleneck: Rappresentazione latente compressa al massimo livello di astrazione.
 3. Decoder (Expanding Path): Ricostruisce la risoluzione originale.
 4. Skip Connections: Uniscono i dettagli spaziali dell'encoder con la semantica 
    del decoder per una localizzazione precisa dei pixel.

STABILITÀ:
- Keras 3 con backend PyTorch per massima flessibilità.
- Ottimizzatore AdamW per regolarizzazione integrata dei pesi.
- Formato .keras per portabilità cross-framework.
================================================================================
"""

import os

# CONFIGURAZIONE AMBIENTE: Impostiamo il backend Keras prima di caricare il modulo.
# Keras 3 è agnostico rispetto al backend (TensorFlow, PyTorch, JAX).
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers

def build_mini_unet(input_shape=(256, 256, 3), num_classes=1):
    """
    Costruisce e restituisce un modello Keras basato sulla U-Net.
    
    Parametri:
    - input_shape: Dimensione dell'immagine in ingresso (Altezza, Larghezza, Canali).
    - num_classes: Numero di maschere in output (1 per segmentazione binaria).
    """
    
    # Definizione dell'Input Layer (Punto di ingresso dei dati nella rete)
    inputs = keras.Input(shape=input_shape)

    # --- 1. ENCODER (Contracting Path) ---
    # Scopo: Ridurre la dimensione spaziale aumentando la profondità (canali).
    # Il modello impara 'COSA' è presente nell'immagine (contesto).

    # Blocco 1: Convoluzioni Doppie (Kernel 3x3, Padding Same per mantenere le dimensioni)
    # layers.Conv2D(64, ...) -> Applica 64 filtri diversi per estrarre bordi e texture.
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    
    # Max Pooling: Riduce le dimensioni (256x256 -> 128x128). 
    # Mantiene solo le attivazioni più forti, rendendo il modello invariante a piccole traslazioni.
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # --- 2. BOTTLENECK ---
    # Punto di massima compressione. Qui la rete ha una visione globale dell'immagine
    # ma ha perso molti dettagli spaziali fini.
    bottleneck = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    bottleneck = layers.Conv2D(128, 3, activation="relu", padding="same")(bottleneck)

    # --- 3. DECODER (Expanding Path) ---
    # Scopo: Ripristinare la dimensione spaziale (Upsampling).
    # Il modello impara 'DOVE' si trovano gli oggetti identificati.

    # UpSampling: Raddoppia la dimensione spaziale (128x128 -> 256x256).
    up1 = layers.UpSampling2D(size=(2, 2))(bottleneck)
    
    # --- 4. SKIP CONNECTION ---
    # INTERAZIONE CRITICA: Concateniamo l'output dell'Encoder (conv1) con il Decoder (up1).
    # Invece di far passare le informazioni solo attraverso il bottleneck (collo di bottiglia),
    # forniamo al decoder i dettagli spaziali perduti direttamente dal primo blocco.
    merge1 = layers.Concatenate()([conv1, up1])
    
    # Blocco Finale di Ricostruzione
    conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")(merge1)
    conv2 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv2)

    # --- 5. OUTPUT LAYER (Pixel-wise Classification) ---
    # Convoluzione 1x1: Mappa i 64 canali nelle classi desiderate.
    # Sigmoid: Produce un valore tra 0 e 1 per ogni pixel (Probabilità di appartenenza alla classe).
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(conv2)

    # Assemblaggio del Modello Finale
    model = keras.Model(inputs=inputs, outputs=outputs, name="Mini_U-Net_Professional")
    return model

# --- ESECUZIONE E COMPILAZIONE ---

# Inizializziamo il modello con i parametri di default
unet_model = build_mini_unet()

# Compilazione: 
# Optimizer AdamW: Evoluzione di Adam con una gestione del Weight Decay più corretta (Standard 2026).
# Loss Binary Crossentropy: Misura l'errore tra la maschera predetta e quella reale a livello di pixel.
# Metric IoU (Intersection over Union): La metrica regina per la segmentazione.
unet_model.compile(
    optimizer="adamw",
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.IoU(num_classes=2, target_class_ids=[1], name="mean_iou")]
)

# Visualizzazione dell'architettura (utile per debuggare i tensori in transito)
unet_model.summary()

# --- VALIDAZIONE E PERSISTENZA ---

# Salvataggio nel formato nativo .keras (Vivamente consigliato rispetto a .h5)
# Permette il caricamento trasparente tra diversi framework ML.
model_path = "semantic_unet_v1.keras"
unet_model.save(model_path)

print(f"\n[INFO] Architettura U-Net configurata correttamente con backend PyTorch.")
print(f"[INFO] Modello salvato con successo in: {model_path}")