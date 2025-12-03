# PsychoAssistantSoul

PsychoAssistantSoul to lekki, samodzielny SDK do komunikacji z **Perplexity Sonar** (OpenAI-compatible endpoint) oraz do wyodrębniania **tagów/keywordów** z tekstu z wykorzystaniem modeli **T5** (Voicelab/vlt5-base-keywords).

Udostępnia prosty interfejs oparty na jednej klasie `ChatbotSDK`, który pozwala:

- generować odpowiedzi oparte na kontekście (Elasticsearch-like JSON),
- automatycznie dostosowywać styl odpowiedzi do typu pytania użytkownika,
- wyciągać słowa kluczowe metodą T5.

## ⚠️ Wymagania

Do działania SDK potrzebny jest **ważny token API Perplexity**:

```
export PERPLEXITY_API_KEY="twoj_token_api"
```

## 📦 Instalacja

```
git clone https://github.com/PsychoAssisant/PsychoAssistantSoul.git
cd PsychoAssistantSoul
pip install -r requirements.txt
```

## 🧠 Podstawowe Użycie

### Inicjalizacja

```python
from PsychoAssistantSoul.sdk import ChatbotSDK
import os

api_key = os.getenv("PERPLEXITY_API_KEY")
bot = ChatbotSDK(api_key=api_key)
```

### Przykład: pytanie bez kontekstu

```python
response = bot.get_response("What are the symptoms of insomnia?")
print(response)
```

### Przykład: użycie z kontekstem

```python
context = [
    {
        "name": "Alpha Clinic",
        "description": "Treatment of sleep disorders",
        "open": "09:00",
        "close": "20:00",
        "paymentMethods": ["CASH", "CARD"]
    }
]

response = bot.get_response(
    "What services does this clinic provide?",
    context=context
)

print(response)
```

## 🏷️ Wyodrębnianie tagów

```python
text = "The patient shows signs of chronic anxiety..."
keywords = bot.extract_keywords(text, max_keywords=3)
print(keywords)
```

## ⚙️ Demo CLI

```
python main.py
```

## 📄 Licencja

MIT
