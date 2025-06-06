import torch
from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..logger import logger
from .anonymizer import Anonymizer


MODEL_PATHS = {
    "length_more": "hallisky/lora-length-long-llama-3-8b",
    "length_less": "hallisky/lora-length-short-llama-3-8b",
    "function_more": "hallisky/lora-function-more-llama-3-8b",
    "function_less": "hallisky/lora-function-less-llama-3-8b",
    "grade_more": "hallisky/lora-grade-highschool-llama-3-8b",
    "grade_less": "hallisky/lora-grade-elementary-llama-3-8b",
    "formality_more": "hallisky/lora-formality-formal-llama-3-8b",
    "formality_less": "hallisky/lora-formality-informal-llama-3-8b",
    "sarcasm_more": "hallisky/lora-sarcasm-more-llama-3-8b",
    "sarcasm_less": "hallisky/lora-sarcasm-less-llama-3-8b",
    "voice_passive": "hallisky/lora-voice-passive-llama-3-8b",
    "voice_active": "hallisky/lora-voice-active-llama-3-8b",
    "type_persuasive": "hallisky/lora-type-persuasive-llama-3-8b",
    "type_expository": "hallisky/lora-type-expository-llama-3-8b",
    "type_narrative": "hallisky/lora-type-narrative-llama-3-8b",
    "type_descriptive": "hallisky/lora-type-descriptive-llama-3-8b",
}
FIRST_MODEL = list(MODEL_PATHS.keys())[0]
MAX_NEW_TOKENS = 1024


def convert_data_to_format(text: str) -> str:
    """
    Converts text to the correct format for LoRA adapters in StyleRemix
    """
    output = f"### Original: {text}\n ### Rewrite:"
    return output


class StyleRemix(Anonymizer):
    def __init__(self, name="StyleRemix"):
        self.name = name
        self.device = "cuda"
        model_id = "meta-llama/Meta-Llama-3-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, add_bos_token=True, add_eos_token=False, padding_side="left"
        )
        self.tokenizer.add_special_tokens({"pad_token": "<padding_token>"})

        self.base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(
            self.device
        )  # device_map="auto" requires accelerate
        self.base_model.resize_token_embeddings(len(self.tokenizer))  # Resize to add pad token. Value doesn't matter
        # Load in the first model
        self.model = PeftModel.from_pretrained(
            self.base_model,
            MODEL_PATHS[FIRST_MODEL],
            adapter_name=FIRST_MODEL,
            torch_dtype=torch.float16,
        ).to(self.device)
        # Load in the rest of the models
        for cur_adapter in MODEL_PATHS.keys():
            if cur_adapter != FIRST_MODEL:
                self.model.load_adapter(MODEL_PATHS[cur_adapter], adapter_name=cur_adapter)

        # print(model.device) # Seems it re-allocates to CPU
        self.model.to(self.device)
        self.model.eval()

        length = 0.5
        function_words = 0.5
        grade_level = 0.5
        formality = 0.5
        sarcasm = 0.5
        voice = 0.5
        persuasive = 0.5
        descriptive = 0
        narrative = 0
        expository = 0

        sliders_dict = {}
        cur_keys = []
        cur_keys.append(
            (
                "length_more" if length > 0 else (None if length == 0 else "length_less"),
                abs(length),
            )
        )
        cur_keys.append(
            (
                "function_more" if function_words > 0 else (None if function_words == 0 else "function_less"),
                abs(function_words),
            )
        )
        cur_keys.append(
            (
                "grade_more" if grade_level > 0 else (None if grade_level == 0 else "grade_less"),
                abs(grade_level),
            )
        )
        cur_keys.append(
            (
                "sarcasm_more" if sarcasm > 0 else (None if sarcasm == 0 else "sarcasm_less"),
                abs(sarcasm),
            )
        )
        cur_keys.append(
            (
                "formality_more" if formality > 0 else (None if formality == 0 else "formality_less"),
                abs(formality),
            )
        )
        cur_keys.append(
            (
                "voice_active" if voice > 0 else (None if voice == 0 else "voice_passive"),
                abs(voice),
            )
        )
        cur_keys.append(("type_persuasive" if persuasive != 0 else None, abs(persuasive)))
        cur_keys.append(("type_descriptive" if descriptive != 0 else None, abs(descriptive)))
        cur_keys.append(("type_narrative" if narrative != 0 else None, abs(narrative)))
        cur_keys.append(("type_expository" if expository != 0 else None, abs(expository)))

        for cur_key in cur_keys:
            if cur_key[0] is not None:
                sliders_dict[cur_key[0]] = cur_key[1]

        # Make the adapter and switch to it
        logger.debug(sliders_dict)

        combo_adapter_name = ""
        for slider_key in sliders_dict:
            logger.debug(slider_key)
            logger.debug(sliders_dict[slider_key])
            combo_adapter_name += slider_key + str(int(100 * sliders_dict[slider_key])) + "-"
        combo_adapter_name = combo_adapter_name[:-1]
        logger.debug(combo_adapter_name)
        logger.debug(list(sliders_dict.values()))
        logger.debug(list(sliders_dict.keys()))
        logger.debug(list(self.model.peft_config.keys()))

        # Add and set the weighted adapater
        self.model.add_weighted_adapter(
            list(sliders_dict.keys()),
            weights=list(sliders_dict.values()),
            adapter_name=combo_adapter_name,
            combination_type="cat",
        )
        self.model.set_adapter(combo_adapter_name)

    def anonymize(self, text: str) -> str:
        # Convert the list of strings in data to a list of model inputs
        converted_text = convert_data_to_format(text)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer(converted_text, return_tensors="pt", max_length=2048, truncation=True).to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, top_p=0.95)
        response = self.tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()

        return response
