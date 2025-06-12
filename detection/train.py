import os
import copy
import random
import numpy as np
import pandas as pd
import argparse

import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from utils import AnomalyDataset
from detection import AnomalyDetector

idx2label = {
    0: 'normal',
    1: 'problem'
}
label2idx = {
    'normal': 0,
    'problem': 1
}

parser = argparse.ArgumentParser(description="Drone Log Analyzer")
    
# Required arguments
parser.add_argument("--feature_col", required=True, default="sentence", help="Level of analysis")

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


def main():
    # Set global seed for reproducibility
    set_seed()
    # Set device (GPU if available, else CPU)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = pd.read_excel('dataset_sentence_labeled.xlsx').sort_values(by='label', ascending=False).drop_duplicates(subset=args.feature_col, keep='first')
    dataset["label"] = dataset['label'].map(label2idx)
    # dataset["multiclass_label"] = dataset['label'].map(label2idx)

    model_name_path = f"swardiantara/{args.feature_col}-problem-embedding"
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    embedding_model = AutoModel.from_pretrained(model_name_path).to(device)

    # Define the custom dataset and dataloaders
    max_seq_length = 64
    batch_size = 64
    num_epochs = 10

    train_dataset = AnomalyDataset(dataset, args.feature_col, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AnomalyDetector(embedding_model, tokenizer).to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        train_loss_epoch = total_train_loss / len(train_loader)
        print(f"{epoch+1}/{num_epochs}: train_loss: {train_loss_epoch}/{total_train_loss}")
    best_model_state = copy.deepcopy(model.state_dict())
    # Save the model
    torch.save(best_model_state, f'{args.feature_col}_pytorch_model.pt')

    return exit(0)


if __name__ == "__main__":
    main()