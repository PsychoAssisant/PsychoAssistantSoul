from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
import json

class ChatbotSDK:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model_name_base_words = "Voicelab/vlt5-base-keywords"
        self.tokenizer_base_words = T5Tokenizer.from_pretrained(self.model_name_base_words)
        self.model_base_words = T5ForConditionalGeneration.from_pretrained(self.model_name_base_words)

    def get_response(self, user_input: str, context: dict) -> str:
        # Konwersja kontekstu (dict -> JSON string)
        context_json = json.dumps(context, ensure_ascii=False)

        # Tworzymy prompt: kontekst + input użytkownika
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