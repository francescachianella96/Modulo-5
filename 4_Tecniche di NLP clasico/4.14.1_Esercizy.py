def calculate_f1(precision, recall):
    """Calcola l'F1-Score data precision e recall."""
    if (precision + recall) == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# --- PUNTO 1: CALCOLO MANUALE F1-SCORE ---
p = 0.90
r = 0.40
f1_risultato = calculate_f1(p, r)

print("-" * 30)
print(f"ESERCIZIO: Calcolo F1-Score")
print(f"Precision: {p}")
print(f"Recall: {r}")
print(f"F1-Score calcolato: {f1_risultato:.4f}")
print("-" * 30)

# --- PUNTO 2: SIMULAZIONE CAMBIO SOGLIA (Esempio Concettuale) ---
def simulate_threshold_change():
    print("\nLOGICA DELLE SOGLIE (Threshold):")
    print("Soglia standard (0.5): Equilibrio tra Precision e Recall.")
    print("Soglia bassa (0.1): Saresti molto più propenso a chiamare 'Critico' anche i dubbi.")
    print("-> Aumenta la RECALL (non perdi i casi veri).")
    print("-> Diminuisce la PRECISION (molti falsi allarmi).")

simulate_threshold_change()

# --- PUNTO 3: CASO ALLARME ANTINCENDIO ---
def fire_alarm_explanation():
    print("\nPERCHÉ RECALL = 1.0 NELL'ANTINCENDIO?")
    explanation = (
        "In un sistema antincendio, il costo di un Falso Negativo (incendio vero non rilevato) "
        "è infinitamente superiore al costo di un Falso Positivo (falso allarme).\n"
        "Recall = 1.0 garantisce che OGNI incendio venga rilevato.\n"
        "Precision bassa significa solo che ogni tanto dovremo uscire dall'edificio per nulla, "
        "ma saremo sicuri di non restare dentro durante un incendio vero."
    )
    print(explanation)

fire_alarm_explanation()