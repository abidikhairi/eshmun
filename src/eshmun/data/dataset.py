import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


class SwissProtDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        min_length: int = 10,
        max_length: int = 400,
    ):

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length

        self.sequences = self._load_sequences()

    def _load_sequences(self):
        seqs = []
        with open(self.file_path, "r") as f:
            for line in f:
                seq = line.strip()
                if len(seq) >= self.min_length and len(seq) <= self.max_length:
                    seqs.append(seq)

        return seqs

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sequence = self.sequences[index]
        tokens = self.tokenizer.encode(f"{self.tokenizer.bos_token}" + sequence + f"{self.tokenizer.eos_token}")
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {"input_ids": input_ids, "labels": labels}

class SwissProtCollator:

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]
        labels = [torch.tensor(ex["labels"]) for ex in batch]
    
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

