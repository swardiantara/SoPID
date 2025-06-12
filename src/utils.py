
import os
import re

import torch
from torch.utils.data import Dataset


class AnomalyDataset(Dataset):
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


def get_latest_folder(directory='.'):
    # Get all folders in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    # Filter folders matching the pattern YYYYMMDD_HHmmss
    pattern = re.compile(r'^\d{8}_\d{6}$')
    matching_folders = [f for f in folders if pattern.match(f)]
    
    # Sort by name (which effectively sorts by date and time given the format)
    matching_folders.sort()
    
    if matching_folders:
        return os.path.join(directory, matching_folders[-1])
    else:
        return None
    

def get_device(model):
    for param in model.parameters():
        return param.device
    for buffer in model.buffers():
        return buffer.device
    raise ValueError("Model has no parameters or buffers on any device.")