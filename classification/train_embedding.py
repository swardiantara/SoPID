import os

import torch
import argparse
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

parser = argparse.ArgumentParser(description="Drone Log Analyzer")
    
# Required arguments
parser.add_argument("--feature_col", required=True, default="sentence", help="Level of analysis")
parser.add_argument("--label_col", required=True, default="problem_type", help="Level of analysis")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training using device: {device}')

# Step 1: Load a pre-trained model
model_name = 'all-MiniLM-L6-v2'  # or 'hkunlp/instructor-xl'
model = SentenceTransformer(model_name).to(device)

# Load your dataset
df = pd.read_excel('dataset_problem_type.xlsx')
args = parser.parse_args()
# Create pairs for contrastive learning
def create_pairs(df, input_col, label_column):
    examples = []
    for label in df[label_column].unique():
        cluster_df = df[df[label_column] == label]
        other_df = df[df[label_column] != label]
        for i, row in cluster_df.iterrows():
            for j, other_row in cluster_df.iterrows():
                if i != j:
                    examples.append(InputExample(texts=[row[input_col], other_row[input_col]], label=1.0))
            for j, other_row in other_df.iterrows():
                examples.append(InputExample(texts=[row[input_col], other_row[input_col]], label=0.0))
    return examples
# def create_pairs(df):
#     examples = []
#     for label in df['cluster_id'].unique():
#         cluster_df = df[df['cluster_id'] == label]
#         other_df = df[df['cluster_id'] != label]
#         for i, row in cluster_df.iterrows():
#             for j, other_row in cluster_df.iterrows():
#                 if i != j:
#                     examples.append(InputExample(texts=[row['message'], other_row['message']], label=1.0))
#             for j, other_row in other_df.iterrows():
#                 examples.append(InputExample(texts=[row['message'], other_row['message']], label=0.0))
#     return examples

examples = create_pairs(df, args.feature_col, args.label_col)
# Step 3: Create DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=64)
# print(train_dataloader)

# Step 4: Define the contrastive loss
train_loss = losses.ContrastiveLoss(model=model)

# Optional: Define evaluator for validation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples)

# Step 5: Train the model
num_epochs = 1
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_path = os.path.join('embeddings', f'{args.feature_col}-{args.label_col}')
os.makedirs(output_path, exist_ok=True)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path
)

# Save the model
model.save(output_path, 'drone-problem-type')
