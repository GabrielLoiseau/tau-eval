import os
from multiprocessing import Pool

from litellm import completion

from ..logger import logger
from .anonymizer import Anonymizer


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


class LLMAuthorship(Anonymizer):
    def __init__(self, model_name="meta-llama/llama-3.2-1b-instruct:free"):
        self.model_name = model_name
        self.name = f"{model_name}/authorship"
        self.max_tokens = 512
        self.system_prompt = """
        You are an efficient assistant.
        """

        self.user_prompt = """
        Rewrite the text to significantly alter its style, tone, and word
        choice while preserving the original meaning. Use figurative or
        descriptive language, vary sentence structures, adjust tone (e.g.,
        formal to conversational), and employ unique synonyms. Avoid retaining
        distinctive stylistic markers. Respond only with the transformed text.
        """

    def anonymize(self, text: str, n_retries: int = 3) -> str:
        provider = f"openrouter/{self.model_name}" if "ollama" not in self.model_name else self.model_name
        try:
            response = completion(
                provider,
                [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt + "\n Original text: " + text + "\n Rewritten text:",
                    },
                ],
                max_tokens=self.max_tokens,
                api_key=OPENROUTER_API_KEY,
                num_retries=n_retries,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"🔴 {e}")
            # Model did not work, returning original text
            return text

    def anonymize_batch(self, texts: list[str], n_jobs: int = 4) -> list[str]:
        with Pool(processes=n_jobs) as P:
            anonymized_texts = P.map(self.anonymize, texts)
        return anonymized_texts


class LLMPII(Anonymizer):
    def __init__(self, model_name="meta-llama/llama-3.2-1b-instruct:free"):
        self.model_name = model_name
        self.name = f"{model_name}/pii"
        self.max_tokens = 512
        self.system_prompt = """
        You are an efficient assistant.
        """

        self.user_prompt = """
        Anonymize user-submitted text by replacing personally identifiable
        information (PII) like names, dates, locations, and professions with
        alternate values. Ensure the rewritten text is natural, coherent,
        and contextually consistent while preserving the original meaning and
        tone. Respond only with the transformed text.
        """

    def anonymize(self, text: str, n_retries: int = 3) -> str:
        provider = f"openrouter/{self.model_name}" if "ollama" not in self.model_name else self.model_name
        try:
            response = completion(
                provider,
                [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt + "\n Original text: " + text + "\n Rewritten text:",
                    },
                ],
                # response_format=Text,
                max_tokens=self.max_tokens,
                api_key=OPENROUTER_API_KEY,
                num_retries=n_retries,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"🔴 {e}")
            # Model did not work, returning original text
            return text

    def anonymize_batch(self, texts: list[str], n_jobs: int = 4) -> list[str]:
        with Pool(processes=n_jobs) as P:
            anonymized_texts = P.map(self.anonymize, texts)
        return anonymized_texts
