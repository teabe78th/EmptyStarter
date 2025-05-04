"""
Moduł zaawansowanej analizy NLP do generowania bardziej kontekstowych
i spersonalizowanych pytań terapeutycznych.

Ten moduł zapewnia dostęp do zaawansowanych modeli NLP, które mogą
lepiej zrozumieć kontekst rozmowy i odpowiednio generować pytania.
"""

import os
import logging
import json
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache na wyniki analizy, aby ograniczyć liczbę zapytań do API
question_cache = {}

# Domyślne pytania, gdy API zawiedzie lub brak kontekstu
DEFAULT_QUESTIONS = [
    "Co sprawiło Ci największą satysfakcję w ostatnim tygodniu?",
    "Jak opisałbyś/opisałabyś swój nastrój w ostatnich dniach?",
    "Jakie emocje towarzyszą Ci najczęściej w ciągu dnia?",
    "Co przede wszystkim motywuje Cię do działania?",
    "Jakie wartości są dla Ciebie najważniejsze w życiu?"
]

# Dostępne modele NLP
class NLPModels:
    CLAUDE_LATEST = "claude-3-5-sonnet-20241022"  # Najnowszy model Claude
    GPT4_LATEST = "gpt-4o"  # Najnowszy model OpenAI
    CLAUDE_LEGACY = "claude-3-opus-20240229"  # Starszy, ale bardziej zaawansowany model Claude

# Inicjalizacja API klientów
HAS_ANTHROPIC = False
anthropic_client = None

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
    
    # Inicjalizacja klienta Anthropic Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        anthropic_client = Anthropic(api_key=api_key)
        logger.info("Zainicjalizowano klienta Advanced NLP z Anthropic API")
    else:
        logger.warning("Brak klucza API Anthropic (ANTHROPIC_API_KEY)")
except ImportError as e:
    logger.warning(f"Nie można zaimportować pakietu Anthropic: {str(e)}")
except Exception as e:
    logger.warning(f"Anthropic API jest niedostępne: {str(e)}")

# Inicjalizacja OpenAI
HAS_OPENAI = False
openai_client = None

try:
    from openai import OpenAI
    HAS_OPENAI = True
    
    # Inicjalizacja klienta OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        logger.info("Zainicjalizowano klienta Advanced NLP z OpenAI API")
    else:
        logger.warning("Brak klucza API OpenAI (OPENAI_API_KEY)")
except Exception as e:
    logger.warning(f"OpenAI API jest niedostępne: {str(e)}")

def analyze_emotional_state(context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analizuje stan emocjonalny użytkownika na podstawie kontekstu rozmowy.
    
    Args:
        context: Lista poprzednich elementów konwersacji.
               Każdy element to słownik z kluczami 'question', 'response' i 'date'.
    
    Returns:
        Dict z analizą stanu emocjonalnego, zawierającą:
            - dominant_emotions (list): Dominujące emocje
            - emotional_state (str): Ogólny stan emocjonalny
            - suggested_focus_areas (list): Sugerowane obszary do skupienia się
    """
    if not context or len(context) < 2:
        return {
            "dominant_emotions": [],
            "emotional_state": "neutral",
            "suggested_focus_areas": ["samoświadomość", "refleksja"]
        }
    
    # Przygotowanie tekstu do analizy
    conversation_text = ""
    for item in context[-5:]:  # Użyj tylko ostatnich 5 wpisów
        if item.get("question") and item.get("response"):
            conversation_text += f"Pytanie: {item['question']}\n"
            conversation_text += f"Odpowiedź: {item['response']}\n\n"
    
    # Cache'owanie na podstawie zawartości rozmowy
    cache_key = hash(conversation_text)
    if cache_key in question_cache:
        logger.info("Używam zbuforowanej analizy emocjonalnej")
        return question_cache[cache_key]
    
    # Domyślna analiza na wypadek błędów
    default_analysis = {
        "dominant_emotions": ["refleksyjność"],
        "emotional_state": "neutral",
        "suggested_focus_areas": ["samoświadomość", "refleksja"]
    }
    
    # Spróbuj użyć Claude do analizy (najlepszy wybór)
    if HAS_ANTHROPIC and anthropic_client:
        try:
            system_prompt = """
            Jesteś psychologiem specjalizującym się w analizie emocjonalnej. Przeanalizuj podaną 
            konwersację terapeutyczną i określ dominujące emocje, ogólny stan emocjonalny 
            oraz sugerowane obszary, na których powinna skupić się dalsza rozmowa.
            
            Zwróć odpowiedź jako JSON w następującym formacie:
            {
                "dominant_emotions": ["emocja1", "emocja2", ...],
                "emotional_state": "ogólny stan (np. positive, negative, mixed, neutral)",
                "suggested_focus_areas": ["obszar1", "obszar2", ...]
            }
            """
            
            logger.info("Próba analizy emocjonalnej z Claude")
            
            message = anthropic_client.messages.create(
                model=NLPModels.CLAUDE_LATEST,
                max_tokens=300,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": conversation_text}
                ]
            )
            
            content = message.content[0].text.strip()
            
            # Wydobycie JSON z odpowiedzi
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            
            analysis = json.loads(content)
            
            # Sprawdzenie, czy mamy wszystkie wymagane klucze
            required_keys = ["dominant_emotions", "emotional_state", "suggested_focus_areas"]
            if all(key in analysis for key in required_keys):
                # Dodaj do cache
                question_cache[cache_key] = analysis
                return analysis
        
        except Exception as e:
            logger.error(f"Błąd podczas analizy emocjonalnej z Claude: {str(e)}")
            # Kontynuuj do OpenAI lub fallbacku
    
    # Jako backup użyj OpenAI
    if HAS_OPENAI and openai_client:
        try:
            system_prompt = """
            Jesteś psychologiem specjalizującym się w analizie emocjonalnej. Przeanalizuj podaną 
            konwersację terapeutyczną i określ dominujące emocje, ogólny stan emocjonalny 
            oraz sugerowane obszary, na których powinna skupić się dalsza rozmowa.
            
            Zwróć odpowiedź jako JSON w następującym formacie:
            {
                "dominant_emotions": ["emocja1", "emocja2", ...],
                "emotional_state": "ogólny stan (np. positive, negative, mixed, neutral)",
                "suggested_focus_areas": ["obszar1", "obszar2", ...]
            }
            """
            
            logger.info("Próba analizy emocjonalnej z OpenAI")
            
            response = openai_client.chat.completions.create(
                model=NLPModels.GPT4_LATEST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Sprawdzenie, czy mamy wszystkie wymagane klucze
            required_keys = ["dominant_emotions", "emotional_state", "suggested_focus_areas"]
            if all(key in analysis for key in required_keys):
                # Dodaj do cache
                question_cache[cache_key] = analysis
                return analysis
                
        except Exception as e:
            logger.error(f"Błąd podczas analizy emocjonalnej z OpenAI: {str(e)}")
    
    # Jeśli wszystko zawiedzie, użyj domyślnej analizy
    return default_analysis

def generate_advanced_question(context: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Generuje zaawansowane pytanie terapeutyczne z wykorzystaniem najnowszych modeli NLP.
    
    Args:
        context: Lista poprzednich elementów konwersacji.
               Każdy element to słownik z kluczami 'question', 'response' i 'date'.
    
    Returns:
        Tuple zawierający:
            - wygenerowane pytanie (str)
            - metadane o generowaniu (dict)
    """
    # Jeśli nie ma kontekstu, wygeneruj pytanie inicjujące
    if not context or len(context) == 0:
        if random.random() < 0.7 and (HAS_ANTHROPIC or HAS_OPENAI):
            initial_question = _generate_initial_question()
            return initial_question, {"model": "advanced", "context_used": False}
        else:
            # Losowe pytanie z domyślnych
            return random.choice(DEFAULT_QUESTIONS), {"model": "default", "context_used": False}
    
    # Analizuj stan emocjonalny na podstawie kontekstu
    emotional_analysis = analyze_emotional_state(context)
    
    # Przygotowanie kontekstu rozmowy
    conversation_text = ""
    for item in context[-5:]:  # Użyj tylko ostatnich 5 wpisów
        if item.get("question") and item.get("response"):
            conversation_text += f"Pytanie: {item['question']}\n"
            conversation_text += f"Odpowiedź: {item['response']}\n"
            conversation_text += f"Data: {item['date']}\n\n"
    
    # Cache'owanie na podstawie zawartości rozmowy
    cache_key = hash(conversation_text)
    if cache_key in question_cache:
        logger.info("Używam zbuforowanego pytania")
        cached_question = question_cache[cache_key]
        return cached_question, {"model": "cached", "context_used": True, "emotional_analysis": emotional_analysis}
    
    # Dostępne stany emocjonalne i sugerowane typy pytań
    question_strategies = {
        "positive": "Zadaj pytanie, które zachęci do refleksji nad pozytywnymi aspektami życia lub doświadczeniami.",
        "negative": "Zadaj empatyczne pytanie, które pomoże w analizie trudnych emocji, ale z perspektywą konstruktywnego rozwiązania.",
        "mixed": "Zadaj pytanie, które pozwoli na zrównoważenie sprzecznych emocji i znalezienie harmonii.",
        "neutral": "Zadaj pytanie, które zgłębi tematy ważne dla osobistego rozwoju i samoświadomości."
    }
    
    # Domyślna strategia
    emotion_strategy = question_strategies.get(
        emotional_analysis.get("emotional_state", "neutral"), 
        question_strategies["neutral"]
    )
    
    # Obszary sugerowane do skupienia się
    focus_areas = emotional_analysis.get("suggested_focus_areas", ["samoświadomość"])
    focus_areas_text = ", ".join(focus_areas)
    
    generated_question = _generate_contextual_question(
        conversation_text, 
        emotion_strategy, 
        focus_areas_text
    )
    
    # Dodaj do cache
    if generated_question:
        question_cache[cache_key] = generated_question
    
    return generated_question or random.choice(DEFAULT_QUESTIONS), {
        "model": "advanced", 
        "context_used": True, 
        "emotional_analysis": emotional_analysis
    }

def _generate_initial_question() -> str:
    """Generuje pierwsze pytanie terapeutyczne bez kontekstu wcześniejszej rozmowy."""
    
    prompt = """
    Wygeneruj jedno głębokie, refleksyjne pytanie terapeutyczne w języku polskim, 
    które mogłoby rozpocząć rozmowę z nowym użytkownikiem aplikacji wsparcia psychologicznego.
    
    Pytanie powinno być empatyczne, otwarte i zachęcające do głębszej refleksji nad sobą
    i swoim samopoczuciem.
    
    Unikaj pytań zamkniętych i powierzchownych. Pytanie powinno być napisane w drugiej
    osobie liczby pojedynczej (Ty).
    
    Odpowiedz tylko samym pytaniem, bez dodatkowego tekstu.
    """
    
    # Strategia 1: Użyj Claude jeśli dostępny (preferowany)
    if HAS_ANTHROPIC and anthropic_client:
        try:
            message = anthropic_client.messages.create(
                model=NLPModels.CLAUDE_LATEST,
                max_tokens=150,
                temperature=0.7,
                system="Jesteś empatycznym polskim psychoterapeutą specjalizującym się w terapii poznawczo-behawioralnej i refleksyjnym podejściu do problemów życiowych.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Błąd podczas generowania pytania inicjującego z Claude: {str(e)}")
    
    # Strategia 2: Użyj OpenAI jako backup
    if HAS_OPENAI and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=NLPModels.GPT4_LATEST,
                messages=[
                    {"role": "system", "content": "Jesteś empatycznym polskim psychoterapeutą specjalizującym się w zadawaniu pytań, które skłaniają do głębokiej refleksji."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania pytania inicjującego z OpenAI: {str(e)}")
    
    # Strategia 3: Użyj domyślnego pytania, jeśli wszystko zawiedzie
    return random.choice(DEFAULT_QUESTIONS)

def _generate_contextual_question(conversation_text: str, emotion_strategy: str, focus_areas: str) -> Optional[str]:
    """
    Generuje kontekstowe pytanie na podstawie analizy rozmowy i stanu emocjonalnego.
    
    Args:
        conversation_text: Pełny tekst poprzednich konwersacji
        emotion_strategy: Strategia pytania dopasowana do stanu emocjonalnego
        focus_areas: Sugerowane obszary do skupienia się
    
    Returns:
        Wygenerowane pytanie lub None w przypadku błędu
    """
    
    system_prompt = f"""
    Jesteś doświadczonym polskim psychoterapeutą prowadzącym terapeutyczną rozmowę.
    Twoim zadaniem jest wygenerowanie pojedynczego, głębokiego pytania w języku polskim,
    które będzie kontynuacją rozmowy z pacjentem.
    
    Na podstawie dostarczonego fragmentu rozmowy, stwórz pytanie, które:
    1. Bezpośrednio odnosi się do tematów poruszonych przez pacjenta
    2. {emotion_strategy}
    3. Skupia się na jednym lub więcej z następujących obszarów: {focus_areas}
    4. Jest sformułowane w sposób otwarty (nie może być odpowiedzią tak/nie)
    5. Nie zawiera osądów ani założeń
    6. Jest empatyczne i pełne zrozumienia
    
    Wygeneruj wyłącznie jedno pytanie, bez wprowadzenia ani wyjaśnień.
    """
    
    # Strategia 1: Użyj Claude jeśli dostępny (preferowany)
    if HAS_ANTHROPIC and anthropic_client:
        try:
            message = anthropic_client.messages.create(
                model=NLPModels.CLAUDE_LATEST,
                max_tokens=200,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": conversation_text}
                ]
            )
            
            return message.content[0].text.strip()
        
        except Exception as e:
            logger.error(f"Błąd podczas generowania kontekstowego pytania z Claude: {str(e)}")
    
    # Strategia 2: Użyj OpenAI jako backup
    if HAS_OPENAI and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model=NLPModels.GPT4_LATEST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Błąd podczas generowania kontekstowego pytania z OpenAI: {str(e)}")
    
    # Jeśli wszystko zawiedzie, zwróć None (caller powinien użyć domyślnego pytania)
    return None