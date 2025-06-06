import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    # This is mainly used for testing purposes.
    def __init__(self, seq_len, vocab_size, dataset_size=100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        return ((torch.rand(self.seq_len + 1, generator=generator) * self.vocab_size).long(),)


class SyntheticDatasetUntokenizedText(Dataset):
    # This is mainly used for testing purposes.
    def __init__(self, seq_len, vocab_size, dataset_size=100, tokenizer='EleutherAI/gpt-neox-20b'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.dataset_size = dataset_size
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(idx)
        out = (torch.rand(self.seq_len + 1, generator=generator) * self.vocab_size).long()
        return self.tokenizer.decode(out)
