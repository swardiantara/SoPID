import os
import sys
import time
import json
import pdfkit
import random
import argparse

import torch
import numpy as np
import pandas as pd

from os import name
from model import ProblemClassifier

from transformers import AutoTokenizer, AutoModel
from captum.attr import LayerIntegratedGradients, visualization


def get_args():
    # Required arguments
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save_model', action='store_true',
                        help="Wether to save model.")
    
    
    args = parser.parse_args()
    return args


label2idx = {
    'Normal': 0,
    'SurroundingEnvironment': 1,
    'HardwareFault': 2,
    'ParamViolation': 3,
    'RegulationViolation': 4,
    'CommunicationIssue': 5,
    'SoftwareFault': 6
}

idx2label = {
    0: 'Normal',
    1: 'SurroundingEnvironment',
    2: 'HardwareFault',
    3: 'ParamViolation',
    4: 'RegulationViolation',
    5: "CommunicationIssue",
    6: 'SoftwareFault'
}

raw2pro = {
    'normal': 'Normal',
    'SurEnv': 'SurroundingEnvironment',
    'HwFlt': 'HardwareFault',
    'ConfIss': 'ParamViolation',
    'VioReg': 'RegulationViolation',
    'CommIss': "CommunicationIssue",
    'Swflt': 'SoftwareFault',
}


class2color = {
    'normal': '#4CAF50',
    'low': '#FFC107',
    'medium': '#FF5722',
    'high': '#FF5722', 
}


def reconstruct_tokens(tokens, attributions):
    words = []
    attribution_score = []
    current_word = ""
    current_attr = 0
    for token, attribution in zip(tokens, attributions):
        if token.startswith("##"):
            current_word += token[2:]  # Remove "##" and append to the current word
            current_attr += attribution
        else:
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token
            current_attr = attribution
    # Append the last word
    if current_word:
        words.append(current_word)
        attribution_score.append(current_attr)

    return words, attribution_score


def infer_pred(model, input_ids, attention_mask):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_prob = torch.softmax(logits, dim=1)  # 'dim' is preferred over 'axis' in PyTorch

        # Get predicted class index (int)
        pred_idx = torch.argmax(pred_prob, dim=1).item()

        # Convert class index to label
        pred_label = idx2label.get(pred_idx, 'Normal')

        # Get predicted probability value (float)
        pred_prob_val = pred_prob[0, pred_idx].item()

    return pred_label, pred_prob_val  # (str, float)


def scale_attribution(distribution):
    """
    Scales the input distribution to the range [-1, 1].

    Parameters:
    distribution (numpy.ndarray): The input distribution of values to be scaled.

    Returns:
    numpy.ndarray: The scaled distribution with values in the range [-1, 1].
    """
    distribution = np.asarray(distribution)
    min_val = np.min(distribution)
    max_val = np.max(distribution)
    scaled_distribution = 2 * (distribution - min_val) / (max_val - min_val) - 1
    return scaled_distribution


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta, vis_data_records):
    attributions = np.array(attributions)
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            word_attributions=attributions,
                            pred_prob=pred,
                            pred_class=pred_ind,
                            true_class=label, # true label
                            attr_class=label, # attribution label
                            attr_score=attributions.sum(),
                            raw_input_ids=text,
                            convergence_score=delta))


vis_data_records_ig = []
def interpret(model: ProblemClassifier, tokenizer: AutoTokenizer, max_seq_length: int, text: str, label: str):
    model.eval()

    labelidx = label2idx.get(label)
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    lig = LayerIntegratedGradients(model, model.embedding_model.embeddings)
    pred_label, pred_prob = infer_pred(model, input_ids, attention_mask)

    attributions, delta = lig.attribute(inputs=input_ids, 
                                        baselines=input_ids*0, 
                                        additional_forward_args=(attention_mask,),
                                        target=labelidx,
                                        return_convergence_delta=True)
    # Sum the attributions across embedding dimensions
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    # Convert attributions to numpy
    attributions = attributions.cpu().detach().numpy()

    # Get the tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens, attributions = reconstruct_tokens(tokens, attributions)
    add_attributions_to_visualizer(attributions, tokens, pred_prob, pred_label, label, delta, vis_data_records_ig)
    return attributions, tokens, label, pred_label, pred_prob


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

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # set global seed for reproducibility
    set_seed(args.seed)
    # set device (GPU if available, else CPU)

    # prepare output directory
    freeze = 'freeze' if args.freeze_embedding else 'unfreeze'
    workdir = os.path.join(args.output_dir, args.feature_col, args.embedding, freeze, str(args.seed))
    print(f'Current scenario: {workdir}')

    print("Starting interpretability report generation...\n")
    time.sleep(2)

    # load the model
    print("Loading the model...")
    model_path = os.path.join(workdir, f'{args.feature_col}_pytorch_model.pt')
    if not os.path.exists(workdir):
        print("The model not found!")
        sys.exit(0)
    
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

    model = ProblemClassifier(embedding_model, tokenizer, freeze_embedding=args.freeze_embedding).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully...")

    # load the test set
    print("Loading the test set...")
    evidence_file = os.path.join('dataset', f'test_{args.feature_col}.xlsx')
    if not os.path.isfile(evidence_file):
        print("The test set not found!")
        sys.exit(0)

    test_set = pd.read_excel(os.path.join('dataset', f'test_sentence.xlsx'))
    test_set["label"] = test_set['problem_type'].map(raw2pro)

    print("Test set loaded successfully...")

    print("Start interpreting...")
    attribution_list = []
    for index, row in test_set.iterrows():
        attributions, tokens, label, pred_label, pred_prob = interpret(model, tokenizer, max_seq_length, row[args.feature_col], row['label'])
        attribution_list.append({
            "index": index + 1,
            "words": tokens,
            "attributions": attributions,
            "label": label,
            "pred_label": idx2label.get(pred_label, 'Normal'),
            "pred_prob": pred_prob,
        })
    html_output = visualization.visualize_text(vis_data_records_ig)
    with open(os.path.join(workdir, f'word_importance_{args.feature_col}.html'), 'w') as f:
        f.write(html_output.data)
    with open(os.path.join(workdir, f"attributions_{args.feature_col}.json"), "w", encoding="utf-8") as f:
        json.dump(attribution_list, f, indent=2, ensure_ascii=False)

    if name == 'nt':
        path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    else:
        path_to_wkhtmltopdf = r'/usr/bin/wkhtmltopdf'

    config_wkhtml = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdfkit.from_string(html_output.data, os.path.join(workdir, f'word_importance_{args.feature_col}.pdf'), configuration=config_wkhtml)
    print("Finish interpreting...")

if __name__ == "__main__":
    main()