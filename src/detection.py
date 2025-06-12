import torch
from torch import nn
from typing import List, Dict, Any
from src.data_loader import LogRecord

    
class AnomalyDetector(nn.Module):
    def __init__(self, embedding_model, tokenizer, hidden_dim=128, dropout_rate=0.1, num_class=2, freeze_embedding=False):
        """
        Args:
            embedding_model: A Hugging Face-compatible transformer model with a .forward method
            tokenizer: The corresponding tokenizer
            hidden_dim: Size of the hidden layer in the classifier
            dropout_rate: Dropout probability
            num_class: Number of classes in the output
            freeze_embedding: Whether to freeze the embedding's parameters
        """
        super(AnomalyDetector, self).__init__()
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.idx2label = {
            0: 'normal',
            1: 'problem'
        }
        self.label2idx = {
            'normal': 0,
            'problem': 1
        }

        if hasattr(embedding_model.config, 'hidden_size'):
            self.embedding_dim = embedding_model.config.hidden_size
        else:
            raise ValueError("Could not determine embedding dimension from model config.")
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_class)
        )

        if freeze_embedding:
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def mean_pooling(self, last_hidden_state, attention_mask):
        # attention_mask: [batch_size, seq_len]
        # last_hidden_state: [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [batch_size, seq_len, 1]
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts  # [batch_size, hidden_size]

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        pooled = self.mean_pooling(last_hidden_state, attention_mask)  # [batch_size, hidden_dim]
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        return logits
    
    def get_device(self):
        for param in self.parameters():
            return param.device
        for buffer in self.buffers():
            return buffer.device
        raise ValueError("Model has no parameters or buffers on any device.")
    
    
    def detect(self, records: List[LogRecord], config: dict) -> List[LogRecord]:
        """Identify problem-indicating message/sentence"""
        device = self.get_device(self)
        self.eval()

        for record in records:
            if not record.sentences:
                continue

            # Message-level analysis
            with torch.no_grad():
                inputs = self.tokenizer(record.raw_message, padding=True, truncation=True, return_tensors="pt").to(device)
                logits = self(inputs["input_ids"], inputs["attention_mask"])
                pred_prob = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(pred_prob, dim=-1).item()
                problem_prob = pred_prob[0, 1].item()
                record.message_anomaly = self.idx2label.get(pred_label)
                record.message_prob = problem_prob

                # Sentence-level analysis
                for sentence in record.sentences:
                    with torch.no_grad():
                        inputs = self.tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)
                        logits = self(inputs["input_ids"], inputs["attention_mask"])
                        pred_prob = torch.softmax(logits, dim=-1)
                        pred_label = torch.argmax(pred_prob, dim=-1).item()
                        problem_prob = pred_prob[0, 1].item()
                        record.sentence_anomaly.append(self.idx2label.get(pred_label))
                        record.sentence_prob.append(problem_prob)

        return records
    

class WordFilterDetector():
    # def __init__(self, args: dict):
    #     self.args = args

    # def load_keywords(file_path: str) -> List[str]:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         return [line.strip().lower() for line in f if line.strip()]

    # def keyword_based_problem_identification(logs: List[LogRecord], keyword_file: str) -> None:
    #     problem_keywords = self.load_keywords(keyword_file)

    #     for record in logs:
    #         msg = record.raw_message.lower()
    #         record.anomaly = "anomaly" if any(kw in msg for kw in problem_keywords) else "normal"

    def detect(self, records: List[LogRecord], config: dict) -> List[LogRecord]:
        # message-level analysis
        for record in records:
            msg = record.raw_message.lower()
            record.message_anomaly = "anomaly" if any(kw in msg for kw in config['keywords']) else "normal"

            # Sentence-level analysis
            for i, sentence in enumerate(record.sentences):
                record.sentence_anomaly[i] = "anomaly" if any(kw in sentence for kw in config['keywords']) else "normal"


        return records