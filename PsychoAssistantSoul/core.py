import json
from typing import Union, List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline,
)


class ChatbotSDK:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        # Model konwersacyjny
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Model do wyciągania słów kluczowych (zostawiam, jak u Ciebie)
        self.model_name_base_words = "Voicelab/vlt5-base-keywords"
        self.tokenizer_base_words = T5Tokenizer.from_pretrained(self.model_name_base_words)
        self.model_base_words = T5ForConditionalGeneration.from_pretrained(self.model_name_base_words)

        # NEW: pipeline Question Answering (możesz podmienić model na inny, np. PL)
        # np. "deepset/xlm-roberta-base-squad2" (wielojęzyczny)
        self.qa = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )

    # --- POMOCNICZE: budowa tekstowego contextu z odpowiedzi Elasticsearch ---

    def _build_context_from_es(
        self,
        context: Union[bytes, str, List[Dict], Dict]
    ) -> str:
        """
        Przyjmuje:
        - bytes (surowa odpowiedź z ES),
        - string z JSON-em,
        - listę słowników (już sparsowane),
        - pojedynczy słownik.

        Zwraca tekst po polsku, który pójdzie jako 'context' do QA pipeline.
        """
        # 1. Normalizacja do obiektu Pythona
        if isinstance(context, bytes):
            context = context.decode("utf-8")

        if isinstance(context, str):
            # Zakładamy że to JSON (np. z ES)
            context = json.loads(context)

        # Teraz context to dict albo list[dict]
        if isinstance(context, dict):
            items = [context]
        elif isinstance(context, list):
            items = context
        else:
            # Jak coś dziwnego – po prostu rzutujemy na string
            return str(context)

        # 2. Budowa kontekstu tekstowego
        parts = []
        for clinic in items:
            name = clinic.get("name", "Brak nazwy")
            description = clinic.get("description", "")
            open_time = clinic.get("open", "")
            close_time = clinic.get("close", "")
            payment_methods = clinic.get("paymentMethods", [])

            part = (
                f"Nazwa kliniki: {name}. "
                f"Opis: {description}. "
                f"Godziny otwarcia: od {open_time} do {close_time}. "
                f"Metody płatności: {', '.join(payment_methods)}."
            )
            parts.append(part)

        return "\n".join(parts)

    # --- GŁÓWNA FUNKCJA ODPOWIEDZI ---

    def get_response(self, user_input: str, context: dict | bytes | str | list | None) -> str:
        """
        Jeżeli jest kontekst z Elasticsearch:
        - buduje z niego tekst,
        - używa pipeline QA, żeby odpowiedzieć NA PODSTAWIE TEGO KONTEKSTU.

        Jeżeli kontekstu brak lub jest pusty:
        - fallback do modelu konwersacyjnego DialoGPT.
        """

        # 1. Jeśli mamy kontekst z ES – użyj QA
        if context:
            context_text = self._build_context_from_es(context)

            if context_text.strip():
                qa_answer = self.qa(
                    question=user_input,
                    context=context_text
                )

                # qa_answer ma formę: {"score": ..., "start": ..., "end": ..., "answer": "..."}
                raw_ans = qa_answer.get("answer", "").strip()

                # Prosta interpretacja "tak/nie" dla pytań o BLIK itp.
                # Możesz to dopracować wg potrzeb.
                lower_q = user_input.lower()
                lower_ans = raw_ans.lower()

                # Przykład: pytanie o BLIK
                if "blik" in lower_q:
                    if "blik" in lower_ans:
                        return "Tak, klinika przyjmuje płatności BLIK."
                    else:
                        # Dodatkowy check w samym kontekście na wszelki wypadek:
                        if "blik" in context_text.lower():
                            return "Tak, klinika przyjmuje płatności BLIK."
                        else:
                            return "Nie widzę w kontekście informacji, że klinika przyjmuje płatności BLIK."

                # Domyślnie zwracamy po prostu fragment odpowiedzi modelu QA
                return raw_ans or "Nie potrafię jednoznacznie odpowiedzieć na to pytanie na podstawie podanego kontekstu."

        # 2. Brak sensownego kontekstu – standardowa odpowiedź generatywna
        context_json = json.dumps(context or {}, ensure_ascii=False)

        prompt = f"Kontekst: {context_json}\nUżytkownik: {user_input}\nAsystent:"
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")

        chat_history_ids = self.model.generate(
            input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )

        response_text = self.tokenizer.decode(
            chat_history_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return response_text

    def extract_keywords(self, text, max_keywords=3):
        # przyjęty prefix może być: "Keywords: " albo inny w zależności od modelu
        prefix = "Keywords: "
        input_text = prefix + text.strip()
        inputs = self.tokenizer_base_words(input_text, return_tensors="pt", truncation=True, max_length=512)

        outputs = self.model_base_words.generate(
            inputs["input_ids"],
            max_length=32,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        generated = self.tokenizer_base_words.decode(outputs[0], skip_special_tokens=True)
        # np. wygenerowany: "słowo1, słowo2, słowo3"
        keywords = [w.strip() for w in generated.split(",")]
        return keywords[:max_keywords]

if __name__ == "__main__":
    bot = ChatbotSDK()
    print("SDK gotowy do pracy. Wpisz 'exit', aby zakończyć.")

    # przykładowe użycie:
    # text = "Jak mogę zabezpieczyć dostęp do bazy danych w aplikacji webowej?"
    # print(bot.extract_keywords(text))

    #
    # while True:
    #     user_text = input("Ty: ")
    #     if user_text.lower() == "exit":
    #         break
    #     reply = bot.get_response(user_text)
    #     print("Bot:", reply)