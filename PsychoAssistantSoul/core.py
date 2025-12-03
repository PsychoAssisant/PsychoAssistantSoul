import json
import logging
import os
from typing import Union, List, Dict, Any

from openai import OpenAI
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)

# --- LOGGING CONFIGURATION ---

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ChatbotSDK:
    """
    Chatbot SDK using Perplexity Sonar API (OpenAI-compatible chat completions).

    - Uses Perplexity's /chat/completions endpoint via OpenAI client.
    - Reads API key from PERPLEXITY_API_KEY environment variable.
    - Builds natural-language context from Elasticsearch-like JSON.
    - Detects simple question type (yes/no, list, explanation, other)
      and adjusts the style of the answer.
    - Still provides a T5-based keyword extractor (optional usage).
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "sonar",
    ):
        logger.info("Initializing ChatbotSDK with Perplexity model: %s", model_name)

        self.model_name = model_name

        # api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Environment variable PERPLEXITY_API_KEY is not set. "
                "Create an API key in Perplexity settings and export it "
                "as PERPLEXITY_API_KEY."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai",
        )
        logger.info("Perplexity client configured (OpenAI-compatible).")

        # Optional keyword extraction model (same as you had before)
        self.model_name_base_words = "Voicelab/vlt5-base-keywords"
        logger.info("Loading keyword extraction model: %s", self.model_name_base_words)
        self.tokenizer_base_words = T5Tokenizer.from_pretrained(
            self.model_name_base_words,
            legacy=True,  # keep old behavior, silence warning
        )
        self.model_base_words = T5ForConditionalGeneration.from_pretrained(
            self.model_name_base_words
        )
        logger.info("Keyword extraction model loaded.")

    def _detect_question_type(self, question: str) -> str:
        """
        Very simple question type heuristic:
        - "yes_no"   – question likely expecting yes/no (is/are/do/can/etc.)
        - "list"     – asking to list/enumerate things
        - "why_how"  – asking for explanation (how/why)
        - "other"    – everything else
        """
        q = question.strip().lower()
        logger.debug("Detecting question type for: %s", q)

        if q.startswith(
            ("is ", "are ", "do ", "does ", "can ", "should ", "will ", "would ")
        ):
            logger.debug("Detected question type: yes_no")
            return "yes_no"

        list_keywords = [
            "list",
            "enumerate",
            "what are",
            "which are",
            "give me the list",
            "show me the list",
            "provide the list",
        ]
        if any(kw in q for kw in list_keywords):
            logger.debug("Detected question type: list")
            return "list"

        why_how_keywords = [
            "how ",
            "how do",
            "how does",
            "why ",
            "explain",
            "for what reason",
            "in what way",
        ]
        if any(kw in q for kw in why_how_keywords):
            logger.debug("Detected question type: why_how")
            return "why_how"

        logger.debug("Detected question type: other")
        return "other"

    def _build_context_from_es(
        self,
        context: Union[bytes, str, List[Dict[str, Any]], Dict[str, Any], None]
    ) -> str:
        """
        Normalizes various forms of Elasticsearch-like responses into
        a readable English context string.

        Accepts:
        - bytes (raw JSON),
        - string (raw JSON or plain text),
        - list[dict] (parsed),
        - dict (single item),
        - None.

        Returns:
            str – human-readable context.
        """
        if context is None:
            logger.debug("Context is None – returning empty string.")
            return ""

        logger.debug("Building context from ES. Input type: %s", type(context))

        if isinstance(context, bytes):
            try:
                context = context.decode("utf-8")
                logger.debug("Decoded bytes to UTF-8 string.")
            except Exception as e:
                logger.warning("Failed to decode bytes: %s", e)
                return str(context)

        if isinstance(context, str):
            try:
                parsed = json.loads(context)
                context = parsed
                logger.debug("String successfully parsed as JSON.")
            except Exception:
                logger.info(
                    "String is not valid JSON – using it directly as context text."
                )
                return context

        if isinstance(context, dict):
            items: List[Dict[str, Any]] = [context]
        elif isinstance(context, list) and all(isinstance(x, dict) for x in context):
            items = context  # type: ignore
        else:
            logger.warning(
                "Unexpected context type after normalization (%s) – casting to string.",
                type(context),
            )
            return str(context)

        logger.info("Number of items in context: %d", len(items))

        parts: List[str] = []

        for idx, item in enumerate(items, start=1):
            name = item.get("name") or item.get("clinicName") or item.get("title")
            description = item.get("description")
            open_time = item.get("open") or item.get("openTime")
            close_time = item.get("close") or item.get("closeTime")
            payment_methods = item.get("paymentMethods") or item.get("payments")
            address = item.get("address")
            city = item.get("city")

            lines: List[str] = [f"Result item {idx}:"]

            if name:
                lines.append(f"- Name: {name}.")
            if description:
                lines.append(f"- Description: {description}.")
            if address or city:
                if address and city:
                    lines.append(f"- Address: {address}, {city}.")
                elif address:
                    lines.append(f"- Address: {address}.")
                elif city:
                    lines.append(f"- City: {city}.")
            if open_time or close_time:
                if open_time and close_time:
                    lines.append(f"- Opening hours: from {open_time} to {close_time}.")
                elif open_time:
                    lines.append(f"- Opening hours: from {open_time}.")
                elif close_time:
                    lines.append(f"- Opening hours: until {close_time}.")
            if isinstance(payment_methods, list) and payment_methods:
                lines.append(
                    f"- Payment methods: {', '.join(map(str, payment_methods))}."
                )
            elif isinstance(payment_methods, str):
                lines.append(f"- Payment methods: {payment_methods}.")

            parts.append("\n".join(lines))

        full_context = "\n\n".join(parts)

        max_chars = 4000
        if len(full_context) > max_chars:
            logger.info("Context longer than %d chars – truncating.", max_chars)
            full_context = (
                full_context[:max_chars]
                + "\n\n[Context truncated due to length for the model.]"
            )

        logger.debug("Built context (first 500 chars): %s", full_context[:500])
        return full_context


    def get_response(
        self,
        user_input: str,
        context: Union[Dict[str, Any], bytes, str, List[Dict[str, Any]], None] = None
    ) -> str:
        """
        Main method to generate a response using Perplexity Sonar.

        - If context is provided, it's turned into human-readable text and
          passed together with the question.
        - If no context: the model answers using only the question.

        The answer style is slightly tuned based on the detected question type.
        """
        logger.info("User question: %s", user_input)

        question_type = self._detect_question_type(user_input)
        logger.info("Detected question type: %s", question_type)

        try:
            context_text = (
                self._build_context_from_es(context) if context is not None else ""
            )
        except Exception as e:
            logger.error("Error while building context: %s", e, exc_info=True)
            context_text = ""

        context_text = context_text.strip()
        has_context = bool(context_text)

        if has_context:
            logger.info("Context provided (length: %d chars).", len(context_text))
        else:
            logger.info("No usable context – answering based only on the question.")

        base_instruction = (
            "You are a helpful assistant. Answer the user's question in English clearly and naturally. "
            "The answer should be based solely on context"
            "If the context does not contain the necessary information, explicitly say that you don't know "
            "instead of inventing details."
        )

        if question_type == "yes_no":
            style_instruction = (
                "If possible, start your answer with 'Yes' or 'No' and then add a brief explanation."
            )
        elif question_type == "list":
            style_instruction = (
                "If it makes sense, respond using a short bullet list. Keep each item concise."
            )
        elif question_type == "why_how":
            style_instruction = (
                "Provide a short, understandable explanation in about 2–4 sentences."
            )
        else:
            style_instruction = "Answer concisely in 1–3 short sentences."

        system_prompt = base_instruction + " " + style_instruction

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

        if has_context:
            user_content = f"Context:\n{context_text}\n\nUser question: {user_input}"
        else:
            user_content = f"User question: {user_input}"

        messages.append({"role": "user", "content": user_content})

        logger.debug("Sending request to Perplexity (model: %s)", self.model_name)

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=512,
            )

            answer = completion.choices[0].message.content.strip()
            if not answer:
                logger.warning("Model returned an empty answer.")
                return "I cannot provide a good answer at the moment."

            logger.info("Generated answer (first 200 chars): %s", answer[:200])
            return answer

        except Exception as e:
            logger.error("Error calling Perplexity API: %s", e, exc_info=True)
            return "An error occurred while generating the response."

    def extract_keywords(self, text: str, max_keywords: int = 3) -> List[str]:
        """
        Extracts keywords using a T5-based model (Voicelab/vlt5-base-keywords).
        Returns at most `max_keywords` keyword phrases.
        """
        logger.info("Extracting keywords from text (length: %d chars).", len(text))

        prefix = "Keywords: "
        input_text = prefix + text.strip()

        try:
            inputs = self.tokenizer_base_words(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            outputs = self.model_base_words.generate(
                inputs["input_ids"],
                max_length=32,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            generated = self.tokenizer_base_words.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            keywords = [w.strip() for w in generated.split(",") if w.strip()]
            result = keywords[:max_keywords]

            logger.info("Extracted keywords: %s", result)
            return result

        except Exception as e:
            logger.error("Error during keyword extraction: %s", e, exc_info=True)
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    bot = ChatbotSDK()
    print("SDK ready. Type 'exit' to quit.\n")

    # Example ES-like context
    sample_context = [
        {
            "id": "b5857557-0143-4989-b16b-6c66034efdd7",
            "name": "Alpha Clinic",
            "description": "Treatment of sleep disorders",
            "open": "09:00:00",
            "close": "20:00:00",
            "paymentMethods": ["BLIK", "CASH"],
        },
        {
            "id": "4d8b49d7-d99b-42d8-b4ed-2001558880a0",
            "name": "Theta Clinic",
            "description": "Treatment of depression and mood disorders",
            "open": "08:00:00",
            "close": "20:00:00",
            "paymentMethods": ["BLIK", "CARD", "CASH"],
        },
    ]

    while True:
        try:
            user_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_text.lower().strip() == "exit":
            break

        reply = bot.get_response(user_text, context=sample_context)
        print("Bot:", reply)
