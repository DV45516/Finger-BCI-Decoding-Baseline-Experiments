#!/bin/bash

subject_ids=($(seq 1 21))
sessions=(1 2)
nclasses=(2 3)
modeltypes=("Orig" "Finetune")

for modeltype in "${modeltypes[@]}"; do
    for subject_id in "${subject_ids[@]}"; do
        for session in "${sessions[@]}"; do
            for nclass in "${nclasses[@]}"; do
                python main_model_training.py "$subject_id" "$session" "$nclass" ME "$modeltype"
            done
        done
    done
done