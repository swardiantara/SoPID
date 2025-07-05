#!/bin/bash

# best-performing model from each embedding model
# drone-sbert
python train_sentence.py --embedding drone-sbert --seed 87212562 --feature_col sentence --n_epochs 100
# bert-base-uncased
python train_sentence.py --embedding bert-base-uncased --seed 70681460 --feature_col sentence --n_epochs 100
# modern-bert
python train_sentence.py --embedding modern-bert --seed 70681460 --feature_col sentence --n_epochs 100
# neo-bert
python train_sentence.py --embedding neo-bert --seed 14298463 --feature_col sentence --n_epochs 100
