from dataclasses import dataclass

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm

from .customtask import CustomTask


def extract_non_o_words(tokens, tags):
    """
    Extract words associated with non-"O" tags by merging tokens.

    Args:
        tokens (list of str): The list of tokens.
        tags (list of str): The list of tags corresponding to each token.

    Returns:
        dict: A dictionary where keys are the tags (e.g., "B-NAME", "B-EMAIL")
              and values are the merged words for each tag.
    """
    result = {}
    current_tag = None
    current_word = []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_tag is not None:
                # Save the completed word for the current tag
                result.setdefault(current_tag, []).append("".join(current_word))
                current_word = []
                current_tag = None
        else:
            base_tag = tag.split("-")[-1]  # Get the base tag (e.g., "NAME", "EMAIL")
            prefix = tag.split("-")[0]  # Get the prefix (B or I)

            # If it's a B-tag or a different tag sequence, start a new word
            if prefix == "B" or current_tag != base_tag:
                # If there's a current word, save it
                if current_tag is not None and current_word:
                    result.setdefault(current_tag, []).append("".join(current_word))

                # Reset for new word
                current_tag = base_tag
                current_word = [token.replace("##", "")]
            else:
                # Continue the current word
                current_word.append(token.replace("##", ""))

    # Save any remaining word
    if current_tag is not None and current_word:
        result.setdefault(current_tag, []).append("".join(current_word))

    return result


def dataset_task_preprocessing(dataset_name: str, dataset_size: int = 2500) -> Dataset:
    match dataset_name:
        case "ai4privacy/pii-masking-400k":
            raw_data = load_dataset(dataset_name)
            texts = []
            labels = []
            for split in raw_data:
                for example in raw_data[split].select(range(dataset_size)):
                    if example["language"] == "en":
                        texts.append(example["source_text"])
                        labels.append(extract_non_o_words(example["mbert_tokens"], example["mbert_token_classes"]))
            final_data = Dataset.from_dict({"text": texts, "labels": labels})

            return final_data
        case _:
            raise NotImplementedError


@dataclass
class DeIdentification(CustomTask):
    dataset: Dataset = None
    name: str = ""
    max_rows: int = None

    def __post_init__(self):
        if isinstance(self.dataset, str):
            name = self.dataset
            self.dataset = dataset_task_preprocessing(self.dataset)

            if not self.name:
                self.name = name

    def evaluate(self, new_texts: list[str]) -> dict:
        assert len(self.dataset) == len(new_texts)

        recall_dict = {}
        result_dict = {}
        total_0 = 0
        total_1 = 0
        for i, example in tqdm(enumerate(self.dataset)):
            for entity_name, values in example["labels"].items():
                if values is None:
                    continue
                if entity_name not in recall_dict.keys():
                    recall_dict[entity_name] = [0, 0]
                for e in values:
                    recall_dict[entity_name][1] += 1
                    total_1 += 1
                    if e.lower() in new_texts[i].lower().replace(" ", ""):
                        recall_dict[entity_name][0] += 1
                        total_0 += 1

        for entity_name, values in recall_dict.items():
            result_dict[f"{entity_name}_recall"] = values[0] / values[1]

        result_dict["Total_recall"] = total_0 / total_1

        return result_dict
