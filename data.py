"""DailyDialog 数据加载"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config import CONTEXT_LENGTH, DATA_DIR, EOS_ID


class DialogueDataset(Dataset):
    def __init__(self, split, tokenizer, context_length=CONTEXT_LENGTH, data_dir=DATA_DIR):
        self.tokenizer = tokenizer
        self.context_length = context_length
        path = Path(data_dir) / f"{split}.txt"
        with open(path, encoding="utf-8") as file:
            self.data = [json.loads(line) for line in file if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        eos = [self.tokenizer.eos_token_id]

        reply = self.tokenizer.encode(sample["reply"]) + eos
        if len(reply) > self.context_length:
            reply = reply[: self.context_length]
            reply[-1] = self.tokenizer.eos_token_id

        history = []
        for text in reversed(sample["history"]):
            utterance = self.tokenizer.encode(text) + eos
            if len(utterance) + len(history) + len(reply) > self.context_length:
                break
            history = utterance + history

        input_ids = history + reply
        labels = [-100] * len(history) + reply
        return {"input_ids": input_ids, "labels": labels}


def collate_examples(examples, eos_token_id=EOS_ID):
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length), eos_token_id, dtype=torch.long
    )
    labels = torch.full((len(examples), max_length), -100, dtype=torch.long)

    for row, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[row, :length] = torch.tensor(example["input_ids"])
        labels[row, :length] = torch.tensor(example["labels"])

    return input_ids, labels
