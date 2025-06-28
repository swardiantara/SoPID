import os
import json
import copy
import torch
import random
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from ast import literal_eval
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, f1_score, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

from utils import MessageDataset, visualize_projection
from model import ProblemClassifier

parser = argparse.ArgumentParser(description="Problem type classification")

raw2pro = {
    'normal': 'Normal',
    'SurEnv': 'SurroundingEnvironment',
    'HwFlt': 'HardwareFault',
    'ConfIss': 'ParamViolation',
    'VioReg': 'RegulationViolation',
    'CommIss': 'CommunicationIssue',
    'Swflt': 'SoftwareFault'
}

pro2idx = {
    'Normal': 0,
    'SurroundingEnvironment': 1,
    'HardwareFault': 2,
    'ParamViolation': 3,
    'RegulationViolation': 4,
    'CommunicationIssue': 5,
    'SoftwareFault': 6
}

class_order = [value for _, value in raw2pro.items()]
# Normal	SurroundingEnvironment	HardwareFault	ParamViolation	RegulationViolation	CommunicationIssue	SoftwareFault

def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--feature_col", required=True, default="sentence", help="Level of analysis")
    parser.add_argument('--output_dir', type=str, default='droptc',
                        help="Folder to store the experimental results. Default: droptc")
    parser.add_argument('--embedding', type=str, choices=['bert-base-uncased', 'neo-bert', 'modern-bert', 'all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1', 'drone-sbert'], default='bert-base-uncased', help='Type of Word Embdding used. Default: bert-base-uncased')
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of testtraining iterations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of samples in a batch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--freeze_embedding', action='store_true',
                        help="Wether to freeze the pre-trained embedding's parameter.")
    
    args = parser.parse_args()
    return args


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


def label_mapper(label_list):
        """Inner function that does the actual mapping for one row"""
        binary_vector = [0] * len(class_order)
        for label in label_list:
            if label in pro2idx:
                binary_vector[pro2idx[label]] = 1
        return binary_vector

def class_mapper(binary_vector):
        """Inner function that converts one binary vector to labels"""
        return [class_order[i] for i, val in enumerate(binary_vector) if val == 1]

def main():
    # Set global seed for reproducibility
    args = get_args()
    set_seed(args.seed)
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare output directory
    freeze = 'freeze' if args.freeze_embedding else 'unfreeze'
    workdir = os.path.join(args.output_dir, args.feature_col, args.embedding, freeze, str(args.seed))
    print(f'current scenario: {workdir}')
    os.makedirs(workdir, exist_ok=True)

    train_df = pd.read_excel(os.path.join('dataset', f'train_{args.feature_col}.xlsx'))
    train_df["labels"] = train_df['labels'].apply(literal_eval)
    train_df['labels'] = train_df['labels'].apply(lambda x: [raw2pro.get(item, item) for item in x])
    train_df['labelidx'] = train_df['labels'].map(label_mapper)

    test_df = pd.read_excel(os.path.join('dataset', f'test_{args.feature_col}.xlsx'))
    test_df["labels"] = test_df['labels'].apply(literal_eval)
    test_df['labels'] = test_df['labels'].apply(lambda x: [raw2pro.get(item, item) for item in x])
    test_df['labelidx'] = test_df['labels'].map(label_mapper)
    
    if args.embedding == 'drone-sbert':
        model_name_path = f"swardiantara/{args.feature_col}-problem_type-embedding"
    elif args.embedding == 'neo-bert':
        model_name_path = 'chandar-lab/NeoBERT'
    elif args.embedding == 'modern-bert':
        model_name_path = 'answerdotai/ModernBERT-base'
    else:
        model_name_path = args.embedding
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(model_name_path, trust_remote_code=True).to(device)

    # Define the custom dataset and dataloaders
    max_seq_length = 64
    batch_size = args.batch_size
    num_epochs = args.n_epochs

    train_dataset = MessageDataset(train_df, tokenizer, max_seq_length)
    test_dataset = MessageDataset(test_df, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ProblemClassifier(embedding_model, tokenizer, freeze_embedding=args.freeze_embedding).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_acc_epoch = float('-inf')
    best_f1_epoch = float('-inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training..."):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss_epoch = total_train_loss / len(train_loader)
        print(f"{epoch+1}/{num_epochs}: train_loss: {train_loss_epoch}/{total_train_loss}")

        # Eval on val dataset
        model.eval()
        val_epoch_labels = []
        val_epoch_preds = []
        total_val_loss
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                preds = [[1 if logit >= 0.5 else 0 for logit in logits] for logits in torch.sigmoid(outputs)]
                val_epoch_preds.extend(preds)
                val_epoch_labels.extend(labels.cpu().numpy())
            val_loss_epoch = total_val_loss / len(test_loader)
        print(f"{epoch+1}/{num_epochs}: val_loss: {val_loss_epoch}/{total_val_loss}")
        assert len(val_epoch_labels) == len(val_epoch_preds), f'labels: {len(val_epoch_labels)}, preds: {len(val_epoch_preds)}'
        val_acc_epoch = accuracy_score(val_epoch_labels, val_epoch_preds)
        val_f1 = f1_score(val_epoch_labels, val_epoch_preds, average='weighted')

        # Check if the current epoch is the best
        if (val_f1 > best_f1_epoch and val_acc_epoch > best_acc_epoch) or (val_f1 > best_f1_epoch and val_acc_epoch >= best_acc_epoch) or (val_f1 >= best_f1_epoch and val_acc_epoch > best_acc_epoch):
            best_f1_epoch = val_f1
            best_acc_epoch = val_acc_epoch
            # Save the model's state (weights and other parameters)
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
    
    # eval the best model
    best_model = ProblemClassifier(embedding_model, tokenizer, freeze_embedding=args.freeze_embedding).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    all_labels_multiclass = []
    all_preds_multiclass = []
    all_preds_probs_multiclass = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation..."):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask)
            preds = [[1 if logit >= 0.5 else 0 for logit in logits] for logits in torch.sigmoid(outputs)]
            assert len(labels) == len(preds), f'labels: {len(labels)}, preds: {len(preds)}'
            all_preds_multiclass.extend(preds)
            all_labels_multiclass.extend(labels)
            all_preds_probs_multiclass.extend(torch.sigmoid(outputs).cpu().numpy())
    
    # preds_names = [class_order[i] for i, val in enumerate(all_preds_multiclass) if val == 1]
    # [list(label_encoder.inverse_transform(key)) for key in all_preds_multiclass]
    # label_names = [list(label_encoder.inverse_transform(key)) for key in all_labels_multiclass]
    preds_decoded = all_preds_multiclass
    tests_decoded = all_labels_multiclass
    prediction_df = pd.DataFrame()
    prediction_df["message"] = test_df["message"]
    prediction_df['label_name'] = test_df['labels']
    prediction_df["labelidx"] = test_df['labelidx']
    prediction_df["predidx"] = list(preds_decoded)
    prediction_df['pred_name'] = prediction_df['predidx'].map(class_mapper)
    prediction_df.to_excel(os.path.join(workdir, "prediction.xlsx"), index=False)

    accuracy = accuracy_score(tests_decoded, preds_decoded)
    f1_weighted = f1_score(tests_decoded, preds_decoded, average='weighted')
    evaluation_report = classification_report(
        tests_decoded, preds_decoded, digits=5)
    classification_report_result = classification_report(
        tests_decoded, preds_decoded, digits=5, output_dict=True)
    classification_report_result['macro_avg'] = classification_report_result.pop('macro avg')
    classification_report_result['weighted_avg'] = classification_report_result.pop('weighted avg')
    micro_pre, micro_rec, micro_f1, _ = precision_recall_fscore_support(tests_decoded, preds_decoded, average='micro')
    classification_report_result['micro_avg'] = {
        "precision": micro_pre,
        "recall": micro_rec,
        "f1-score": micro_f1
        }
    classification_report_result['accuracy'] = accuracy
    # Logs the evaluation results into files
    with open(os.path.join(workdir, "evaluation_report.json"), 'w') as json_file:
        json.dump(classification_report_result, json_file, indent=4)
    with open(os.path.join(workdir, "evaluation_report.txt"), "w") as text_file:
        text_file.write(evaluation_report)
    print("Best epoch: ", best_epoch)
    print("Accuracy:", accuracy)
    print("F1-score:", f1_weighted)
    print("Classification Report:\n", evaluation_report)

    arguments_dict = vars(args)
    arguments_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    arguments_dict['scenario_dir'] = workdir
    arguments_dict['best_epoch'] = best_epoch
    arguments_dict['best_val_f1'] = best_f1_epoch
    arguments_dict['best_val_acc'] = best_acc_epoch
    
    with open(os.path.join(workdir, 'scenario_arguments.json'), 'w') as json_file:
        json.dump(arguments_dict, json_file, indent=4)
    
    # generate a confusion matrix visualization to ease analysis
    class_names = [value for _, value in raw2pro.items()]
    cms = multilabel_confusion_matrix(prediction_df['labelidx'].to_list(), prediction_df['predidx'].to_list())
    for i, matrix in enumerate(cms):
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=False, square=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'{class_names[i]}')
        plt.savefig(os.path.join(workdir, f"confusion_matrix_{class_names[i]}.pdf"), bbox_inches='tight')
        plt.close()

    # Save the model's hidden state to a 2D plot
    # visualize_projection(merged_loader, idx2pro, best_model.to(device), device, workdir)
    # Save the model
    # torch.save(best_model_state, os.path.join(workdir, 'sentence_pytorch_model.pt'))

    return exit(0)

if __name__ == "__main__":
    main()