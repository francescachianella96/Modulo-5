import numpy as np

# 1. Definizione vettori (ID concettuale)
v_gatto = np.array([0.9, 0.1])
v_micio_A = np.array([0.85, 0.12]) # Molto vicino a gatto
v_micio_B = np.array([-0.1, -0.9]) # Molto lontano

# 2. Scelta
# Il Vettore A è il più appropriato perché "Micio" è un sinonimo di "Gatto". 
# In uno spazio di embedding, i sinonimi devono avere coordinate quasi identiche.

# 3. Calcolo Distanza Euclidea (Teoria: L2 Norm)
distanza = np.linalg.norm(v_gatto - v_micio_A)

print(f"Distanza semantica tra Gatto e Micio (A): {distanza:.4f}")
# Una distanza piccola (es. 0.0539) conferma la bontà della rappresentazione densa.