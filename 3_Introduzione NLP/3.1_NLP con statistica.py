# --- 1. MOTORE NLU POTENZIATO (Natural Language Understanding) ---
def simple_nlu_engine(text: str):
    """
    Analizza il testo e calcola un punteggio di confidenza per ogni intento.
    Teoria: Passiamo da un matching booleano a un sistema pesato per gestire l'ambiguità.
    """
    # Pre-processing: normalizzazione del testo
    normalized_text = text.lower().strip()
    
    # Mappatura Intenti/Parole Chiave
    intents_map = {
        "meteo": ["tempo", "piove", "sole", "gradi", "previsioni", "caldo", "freddo"],
        "saluto": ["ciao", "buongiorno", "salve", "ehi", "sveglia"],
        "domotica": ["accendi", "spegni", "luce", "lampada", "clima", "casa", "dispositivo"]
    }
    
    # Inizializziamo il dizionario dei punteggi (Confidence Scores)
    scores = {intent: 0 for intent in intents_map.keys()}
    
    # Calcolo delle occorrenze
    words_in_text = normalized_text.split()
    for intent, keywords in intents_map.items():
        for word in words_in_text:
            if word in keywords:
                scores[intent] += 1
                
    # Determiniamo l'intento con il punteggio massimo
    best_intent = "sconosciuto"
    max_score = 0
    
    for intent, score in scores.items():
        if score > max_score:
            max_score = score
            best_intent = intent
            
    # Log diagnostico della confidenza (Best Practice)
    if max_score > 0:
        print(f"[NLU LOG] Score distribuzione: {scores}")
        
    return best_intent

# --- 2. MODULO NLG (Natural Language Generation) ---
responses = {
    "saluto": "Bentornato! Come posso aiutarti oggi?",
    "meteo": "Al momento ci sono 22°C e il cielo è sereno.",
    "domotica": "Comando ricevuto: sto agendo sui dispositivi della casa.",
    "sconosciuto": "Scusami, non ho abbastanza informazioni per agire. Puoi essere più specifico?"
}

def generate_response(intent: str):
    """Mappa l'intento alla risposta pragmatica più coerente."""
    return responses.get(intent, responses["sconosciuto"])


# --- TEST DELLA NUOVA LOGICA (Scenario Ambiguità) ---
# Caso 1: Comando senza verbo (ora funzionante grazie al punteggio)
test_input_1 = "La luce della cucina"
# Caso 2: Comando misto (prevale la domotica per numero di keyword)
test_input_2 = "Ciao, per favore accendi la luce e il clima"

for user_input in [test_input_1, test_input_2]:
    intent_detected = simple_nlu_engine(user_input)
    final_response = generate_response(intent_detected)

    print(f"Input Utente: '{user_input}'")
    print(f"Intento Rilevato: {intent_detected.upper()}")
    print(f"Risposta AI: {final_response}")
    print("-" * 50)