#!/bin/bash
word_embeds=( drone-sbert bert-base-uncased neo-bert modern-bert )
freezes=( true false )
seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

for embedding in "${word_embeds[@]}"; do
    for seed in "${seeds[@]}"; do
        for freeze in "${freezes[@]}"; do
            if [ "$freeze" = true ]; then
                python train_message.py --embedding "$embedding" --seed "$seed" --feature_col message --freeze_embedding 
            else
                python train_message.py --embedding "$embedding" --seed "$seed" --feature_col message
            fi
        done
    done
done