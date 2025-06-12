import pandas as pd
from typing import List
from dataclasses import dataclass, field

@dataclass
class LogRecord:
    date: str
    time: str
    raw_message: str
    message_anomaly: str = field(default_factory=str)
    message_prob: str = field(default_factory=str)
    message_eventId: str = field(default_factory=str)
    sentences: List[str] = field(default_factory=list)            # store the segmented sentences 
    sentence_types: List[str] = field(default_factory=list)       # store the sentence type (Event or NonEvent)
    sentence_eventId: List[str] = field(default_factory=list)             # store the abstracted events with IDs (E## for Event, N## for NonEvent)
    sentence_anomaly: List[str] = field(default_factory=list)            # store the predicted anomaly severity for each sentence
    sentence_prob: List[float] = field(default_factory=list)       # store the prediction probability of the anomaly severity
    # anomaly_probs: List[float] = field(default_factory=list)      # store the prediction probability of the anomaly severity
    # sum_attributions: List[float] = field(default_factory=list)   # store the attribution score towards the class High
    # max_attributions: List[float] = field(default_factory=list)   # store the attribution score towards the class High
    # norm_attributions: List[float] = field(default_factory=list)  # store the attribution score towards the class High
    # top3_attributions: List[float] = field(default_factory=list)   # store the attribution score towards the class High
    # top5_attributions: List[float] = field(default_factory=list)   # store the attribution score towards the class High
    # snr_std: List[float] = field(default_factory=list)         # store the attribution score towards the class High
    # snr_signed: List[float] = field(default_factory=list)         # store the attribution score towards the class High
    # snr_normed: List[float] = field(default_factory=list)         # store the attribution score towards the class High
    # snr_top3: List[float] = field(default_factory=list)         # store the attribution score towards the class High
    # snr_top5: List[float] = field(default_factory=list)         # store the attribution score towards the class High
    # snr_entropy: List[float] = field(default_factory=list)         # store the attribution score towards the class High


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def load_data(self) -> List[LogRecord]:
        """Load CSV and initialize log records"""
        df = pd.read_excel(self.filepath)
        records = []
        for _, row in df.iterrows():
            records.append(LogRecord(
                date=row['date'],
                time=row['time'],
                raw_message=row['message']
            ))
        return records
    
