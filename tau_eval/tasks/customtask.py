from dataclasses import dataclass

from datasets import Dataset


@dataclass
class CustomTask:
    dataset: Dataset = None
    name: str = ""
    s1: str = "text"
    s2: str = ""

    def evaluate(self, new_texts: list[str]) -> dict:
        raise NotImplementedError
