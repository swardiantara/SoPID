import os
import torch

import pandas as pd
import umap.umap_ as umap
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap


class SentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['sentence']
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


class MessageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['message']
        label = self.data.iloc[idx]['labelidx']
        
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
            'label': torch.tensor(label, dtype=torch.float32),
            "labelidx": label,
        }


def visualize_sentence(dataset_loader, idx2label, model, device, output_dir):
    all_labels = []
    all_embeddings = []
    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_index = batch["label"]
    
            _ = model(input_ids, attention_mask)
            all_labels.extend(labels_index.cpu().numpy())
            all_embeddings.append(model.hidden)
    
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    reduced_embeddings = tsne.fit_transform(all_embeddings.cpu().numpy())
    # label_decoded = [idx2label.get(key) for key in all_labels]
    # label_encoder_multi.inverse_transform(all_labels)
    label_df = pd.DataFrame()
    label_df["label"] = all_labels
    label_df["label"] = label_df["label"].map(idx2label)
    labels = label_df['label'].tolist()
    
    plt.figure(figsize=(5, 2.5))
    fig, ax = plt.subplots()

    unique_labels = label_df['label'].unique()
    # colors = ['#4CAF50', '#FFC107', '#FF5722', '#D32F2F']
    
    counter = 0
    for label in unique_labels:
        # Filter data points for each unique label
        x_filtered = [reduced_embeddings[i][0] for i in range(len(reduced_embeddings)) if labels[i] == label]
        y_filtered = [reduced_embeddings[i][1] for i in range(len(reduced_embeddings)) if labels[i] == label]
        ax.scatter(x_filtered, y_filtered, label=label, s=8, cmap='viridis')
        counter+=1

    # Add a legend with only unique labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax.grid(True)
    # plt.legend([]).set_visible(False)
    # plt.legend()
    # Display the plot
    plt.savefig(os.path.join(output_dir, "dataset_viz.pdf"), bbox_inches='tight')
    plt.close()


def reduce_dimension(embeddings, random_state, technique = 'pca-tsne'):
    # PCA
    if technique == 'pca':
        pca = PCA(n_components=2, random_state=random_state)
        reduced_emb = pca.fit_transform(embeddings)

    # PCA and t-SNE
    elif technique == 'pca-tsne':
        pca = PCA(n_components=50, random_state=random_state)
        reduced_pca = pca.fit_transform(embeddings)
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
        reduced_emb = tsne.fit_transform(reduced_pca)

    # t-SNE
    elif technique == 'tsne':
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
        reduced_emb = tsne.fit_transform(embeddings)

    # UMAP
    elif technique == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        reduced_emb = reducer.fit_transform(embeddings)

    # Isomap
    elif technique == 'isomap':
        isomap = Isomap(n_components=2)
        reduced_emb = isomap.fit_transform(embeddings)

    return reduced_emb


def interactive_plot(dataframe: pd.DataFrame, dataset_loader, text_col, label_col, model, device, dim_reduce, random_state, out_dir):
    filename = os.path.join(out_dir, f'interactive_plot_{text_col}_{dim_reduce}.html')
    if os.path.exists(filename):
        print('The graph has been generated. Skipped!')
        return None
    
    all_embeddings = []
    with torch.no_grad():
        for batch in dataset_loader: # make sure that the data is not shuffled by the data loader
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
    
            _ = model(input_ids, attention_mask)
            all_embeddings.append(model.hidden)
    # Create a DataFrame that includes t-SNE features and the original text labels
    embeddings = torch.cat(all_embeddings, dim=0)
    reduced_embeddings = reduce_dimension(embeddings.cpu().numpy(), random_state, dim_reduce)
    drone_2d_df = pd.DataFrame(reduced_embeddings, columns=['t-SNE1', 't-SNE2'])
    drone_2d_df['Label'] = dataframe[label_col]
    if text_col == 'message':
        drone_2d_df['Label'] = drone_2d_df['Label'].apply(lambda x: "-".join(item for item in x))
    drone_2d_df['Text'] = dataframe[text_col]

    # Plot with Plotly
    fig = px.scatter(drone_2d_df, x='t-SNE1', y='t-SNE2', color="Label", hover_data={'Text': True, 'Label': True, 't-SNE1': False, 't-SNE2': False})
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(title='t-SNE Visualization with Original Text Labels')
    fig.write_html(filename)


def visualize_message(dataset_loader, idx2label, model, device, output_dir):
    all_labels = []
    all_embeddings = []
    with torch.no_grad():
        for batch in dataset_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_index = batch["label"]
    
            embeddings = model(input_ids, attention_mask)
            all_labels.extend(labels_index.cpu().numpy())
            all_embeddings.append(embeddings.last_hidden_state)
    
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    reduced_embeddings = tsne.fit_transform(all_embeddings.cpu().numpy())
    # label_decoded = [idx2label.get(key) for key in all_labels]
    # label_encoder_multi.inverse_transform(all_labels)
    label_df = pd.DataFrame()
    label_df["label"] = all_labels
    label_df["label"] = label_df["label"].map(idx2label)
    labels = label_df['label'].tolist()
    
    plt.figure(figsize=(5, 2.5))
    fig, ax = plt.subplots()

    unique_labels = label_df['label'].unique()
    # colors = ['#4CAF50', '#FFC107', '#FF5722', '#D32F2F']
    
    counter = 0
    for label in unique_labels:
        # Filter data points for each unique label
        x_filtered = [reduced_embeddings[i][0] for i in range(len(reduced_embeddings)) if labels[i] == label]
        y_filtered = [reduced_embeddings[i][1] for i in range(len(reduced_embeddings)) if labels[i] == label]
        ax.scatter(x_filtered, y_filtered, label=label, s=15)
        counter+=1

    # Add a legend with only unique labels
    ax.set_xticks([])
    ax.set_yticks([])
    # legend = ax.legend(loc='lower right')
    plt.legend([]).set_visible(False)
    # Display the plot
    plt.savefig(os.path.join(output_dir, "dataset_viz.pdf"), bbox_inches='tight')
    plt.close()