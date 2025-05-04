
import random
import logging
from anthropic import Anthropic
import os

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja klienta Anthropic
anthropic_client = None
try:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        anthropic_client = Anthropic(api_key=api_key)
except Exception as e:
    logger.warning(f"Nie można zainicjalizować Anthropic API: {str(e)}")

# Domyślna pula cytatów
DEFAULT_QUOTES = [
    "Każda podróż zaczyna się od pierwszego kroku.",
    "W ciszy odnajdujemy własne odpowiedzi.",
    "Zmiana zaczyna się od akceptacji tego, co jest.",
    "Twoje myśli kształtują Twoją rzeczywistość.",
    "Uważność to klucz do zrozumienia siebie.",
]

quote_cache = {}

def generate_therapeutic_quote(context=None):
    """
    Generuje lub wybiera terapeutyczny cytat.
    
    Args:
        context (str, optional): Kontekst dla generowania cytatu
        
    Returns:
        str: Terapeutyczny cytat
    """
    if not anthropic_client:
        return random.choice(DEFAULT_QUOTES)
        
    try:
        prompt = """
        Wygeneruj jeden krótki, mądry cytat terapeutyczny w języku polskim.
        Cytat powinien być inspirujący, głęboki i związany z samorozwojem, 
        ale nie dłuższy niż jedno zdanie.
        
        Odpowiedz tylko samym cytatem, bez cudzysłowów czy dodatkowego tekstu.
        """
        
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        quote = message.content[0].text.strip()
        return quote
        
    except Exception as e:
        logger.error(f"Błąd podczas generowania cytatu: {str(e)}")
        return random.choice(DEFAULT_QUOTES)
