#!/bin/bash
word_embeds=( all-MiniLM-L6-v2 all-MiniLM-L12-v2 all-mpnet-base-v2 all-distilroberta-v1 )
freezes=( true false )
seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

for embedding in "${word_embeds[@]}"; do
    for seed in "${seeds[@]}"; do
        for freeze in "${freezes[@]}"; do
            if [ "$freeze" = true ]; then
                python train_model.py --embedding "$embedding" --seed "$seed" --feature_col sentence --freeze_embedding 
            else
                python train_model.py --embedding "$embedding" --seed "$seed" --feature_col sentence
            fi
        done
    done
done