import os
import glob
import random
import joblib

from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training using device: {device}')

# Step 1: Load a pre-trained model
model_name = 'all-MiniLM-L6-v2'  # or 'hkunlp/instructor-xl'
model = SentenceTransformer(model_name).to(device)

# Step 2: Prepare the dataset
def merge_log_datasets(input_dir, output_file, label_column=['Source', 'EventId'], samples_per_event=1):
    """
    Merge multiple log parsing CSV files into a single dataset.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CSV log files
    output_file : str
        Path to save the merged dataset
    samples_per_event : int
        Number of random samples to select for each EventId (excluding the template sample)
    """
    # Get all CSV files in the input directory
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    # Initialize empty list to store the merged data
    merged_data = []
    
    # Process each CSV file
    for file_path in csv_files:
        # Extract source name from filename
        source_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[0]
        print(f"Processing {source_name}...")
        if source_name == 'DroneOvs':
            print('Skip DroneOvs...')
        # Read the CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        # Check if required columns exist
        required_cols = ['Content', 'EventId', 'EventTemplate']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {file_path} - missing required columns")
            continue
            
        # Add source column
        # df['Source'] = source_name
        if len(label_column) > 1:
            df['Label'] = source_name + '_' + df['EventId']
        else:
            df['Label'] = source_name
        
        # Group by EventId
        event_groups = dict(list(df.groupby('Label')))
        samples_per_event = (samples_per_event + 1) if source_name == 'Drone' else samples_per_event
        for event_id, group in event_groups.items():
            # Get event template for this event ID
            event_template = group['EventTemplate'].iloc[0]
            
            # Select random samples
            if len(group) <= samples_per_event:
                # If fewer than required samples, use all available
                selected_samples = group[['Label', 'Content', 'EventId']]
            else:
                # Randomly select samples
                random_indices = random.sample(range(len(group)), samples_per_event)
                selected_samples = group.iloc[random_indices][['Label', 'Content', 'EventId']]
            
            # Add samples to merged data
            merged_data.extend(selected_samples.to_dict('records'))
            
            if source_name != 'Drone':
                # Add the template as an additional sample
                template_sample = {
                    'Label': source_name + '_' + event_id,
                    'Content': event_template,  # Use template as content
                    'EventId': event_id
                }
                merged_data.append(template_sample)
    
    if not merged_data:
        raise ValueError("No valid data found in any of the CSV files")
    
    # Convert to DataFrame
    merged_df = pd.DataFrame(merged_data)
    
    # Keep only required columns
    merged_df = merged_df[['Label', 'Content', 'EventId']]
    
    # Save the merged dataset
    merged_df.to_csv(output_file, index=False)
    print(f"Merged dataset saved to {output_file}")
    print(f"Total events: {merged_df['Label'].nunique()}")
    print(f"Total samples: {len(merged_df)}")
    # print(f"Unique Label: {', '.join(merged_df['Label'].unique())}")
    
    return merged_df

# Load your dataset
# input_directory = "./dataset"  # Directory containing your CSV files
# output_path = "./merged_logs.csv"  # Where to save the merged dataset
# if not os.path.exists('merged_logs.csv'):
#     df = merge_log_datasets(input_directory, output_path)
# else:
#     df = pd.read_csv('merged_logs.csv')
input_feature = 'sentence'
df = pd.read_excel('dataset_sentence_labeled.xlsx').drop_duplicates(subset=[input_feature])
# Create pairs for contrastive learning
# Eq. (1) and Eq. (2)
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
# scenario = 'Merged_1'
# if not os.path.exists(f'sample_pairs_{scenario}.joblib'):
examples = create_pairs(df, input_feature, 'label')
#     joblib.dump(examples, f'sample_pairs_{scenario}.joblib')
# else:
#     examples = joblib.load(f'sample_pairs_{scenario}.joblib')

# Step 3: Create DataLoader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=8)

# Step 4: Define the contrastive loss
# Eq. (4)
train_loss = losses.ContrastiveLoss(model=model, margin=0.3)

# Optional: Define evaluator for validation
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples, name=f'{input_feature}-embedding')

# Step 5: Train the model
num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
output_path = os.path.join('embedding')
if not os.path.exists(output_path):
    os.makedirs(output_path)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_path
)

# Save the model
model.save(output_path, f'{input_feature}-embedding')
