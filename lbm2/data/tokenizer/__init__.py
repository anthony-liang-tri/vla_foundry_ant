import torch
from transformers import AutoTokenizer

class DebugTokenizer:
    def __init__(self):
        self.pad_token = "[PAD]"

    def __call__(self, texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt'):
        return {
            'input_ids': torch.randint(0, 100, (len(texts), max_length)),
            'attention_mask': torch.ones((len(texts), max_length), dtype=torch.long),
        }

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "debug":
        return DebugTokenizer()
    else:
        return AutoTokenizer.from_pretrained(tokenizer_name)