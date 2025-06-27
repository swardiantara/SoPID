import os

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

class SentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['sentence']
        label = self.data.iloc[index]["labelidx"]

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


class MessageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length, label_encoder=None):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_length
        
        # Initialize label encoder
        # if label_encoder is None:
        #     self.label_encoder = MultiLabelBinarizer()
        #     self.labels = self.label_encoder.fit_transform(dataframe['labels'].to_list())
        # else:
        #     self.label_encoder = label_encoder
        #     self.labels = label_encoder.transform(dataframe['labels'].to_list())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['message'])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            "labelidx": self.labels[idx],
        }



def visualize_projection(dataset_loader, idx2label, model, device, output_dir):
    all_labels_multiclass = []
    all_embeddings = []
    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_index = batch["labelidx"]
    
            embeddings = model.embedding_model(input_ids, attention_mask)
            all_labels_multiclass.extend(labels_index.cpu().numpy())
            all_embeddings.append(embeddings.last_hidden_state)
    
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    reduced_embeddings = tsne.fit_transform(all_embeddings.cpu().numpy())
    label_decoded = [idx2label.get(key) for key in all_labels_multiclass]
    # label_encoder_multi.inverse_transform(all_labels_multiclass)
    label_df = pd.DataFrame()
    label_df["label"] = list(label_decoded)
    labels = label_df['label'].tolist()
    
    plt.figure(figsize=(5, 2.5))
    fig, ax = plt.subplots()

    unique_labels = ['Normal', 'Low', 'Medium', 'High']
    colors = ['#4CAF50', '#FFC107', '#FF5722', '#D32F2F']
    
    counter = 0
    for label in unique_labels:
        # Filter data points for each unique label
        x_filtered = [reduced_embeddings[i][0] for i in range(len(reduced_embeddings)) if labels[i] == label]
        y_filtered = [reduced_embeddings[i][1] for i in range(len(reduced_embeddings)) if labels[i] == label]
        ax.scatter(x_filtered, y_filtered, label=label, s=15, c=colors[counter])
        counter+=1

    # Add a legend with only unique labels
    ax.set_xticks([])
    ax.set_yticks([])
    # legend = ax.legend(loc='lower right')
    plt.legend([]).set_visible(False)
    # Display the plot
    plt.savefig(os.path.join(output_dir, "dataset_viz.pdf"), bbox_inches='tight')
    plt.close()