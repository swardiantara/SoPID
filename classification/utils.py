import torch
from torch.utils.data import Dataset

class ProblemDataset(Dataset):
    def __init__(self, dataframe, feature, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature = feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index][self.feature]
        label = self.data.iloc[index]["label"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labelidx": label,
            "label": torch.tensor(label, dtype=torch.long),
        }