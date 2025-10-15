from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatbotSDK:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def get_response(self, user_input: str) -> str:
        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors="pt")
        chat_history_ids = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response_text = self.tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response_text

if __name__ == "__main__":
    bot = ChatbotSDK()
    print("SDK gotowy do pracy. Wpisz 'exit', aby zakończyć.")

    while True:
        user_text = input("Ty: ")
        if user_text.lower() == "exit":
            break
        reply = bot.get_response(user_text)
        print("Bot:", reply)