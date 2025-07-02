#!/bin/bash

# best-performing model from each embedding model
# drone-sbert
python interpretability.py --embedding drone-sbert --seed 87212562 --feature_col sentence
# bert-base-uncased
python interpretability.py --embedding bert-base-uncased --seed 70681460 --feature_col sentence
# modern-bert
python interpretability.py --embedding modern-bert --seed 70681460 --feature_col sentence
# neo-bert
python interpretability.py --embedding neo-bert --seed 14298463 --feature_col sentence
