from .anonymizer import Anonymizer


class DummyModel(Anonymizer):
    def __init__(self):
        self.name = "Dummy Model"

    def anonymize(self, text) -> str:
        return "..."

    def anonymize_batch(self, texts: list[str]) -> list[str]:
        return ["..."] * len(texts)
