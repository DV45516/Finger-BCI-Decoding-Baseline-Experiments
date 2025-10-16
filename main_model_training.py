#   main_model_training.py

#   This is the main deep learning training script used in the manuscript:
#   "EEG-based Brain-Computer Interface Enables Real-time Robotic Hand Control at Individual Finger Level"
#   
#   Takes in 5 arguments:
#   subj_id: (int) the subject ID, 1-21
#   session_num: (int) the session number, 1-5
#   nclass: (int) number of classes, 2 or 3
#   task: (string) motor imagery or execution, "ME" or "MI"
#   modeltype: (string) pre-training or fine-tuning, "Orig" or "Finetune"
#   
#   Example use: python main_model_training.py 1 1 2 ME Orig
#   Copyright (C) Yidan Ding 2025

# %%

from Functions import load_and_filter_data, generate_paths, train_models, evaluate_model
import tensorflow as tf

import os
import sys
import numpy as np
import pandas as pd

# Read command-line arguments
subj_id = int(sys.argv[1])
session_num = int(sys.argv[2])
nclass = int(sys.argv[3])
task = sys.argv[4]
modeltype = sys.argv[5]

# validate inputs
if nclass not in (2, 3):
    raise ValueError("nclass must be either 2 or 3.")

if task not in ("MI", "ME"):
    raise ValueError("task must be either 'MI' (motor imagery) or 'ME' (motor execution).")

if modeltype not in ("Orig", "Finetune"):
    raise ValueError("modeltype must be either 'Orig' (pre-training) or 'Finetune' (fine-tuning).")

# parameters
params = {
    'maxtriallen':5, # in s
    'windowlen':1, # in s
    'block_size': 128, # (samples) same as the online config
    'downsrate': 100, # dawnsampling rate
    'bandpass_filt': [4,40], # (Hz) bandpass filtering
    'nclass': nclass
}
# Specify the paths to data and the models
data_folder = 'pathToData'
save_folder = 'pathToSave'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

data_paths = generate_paths(subj_id, task, nclass, session_num, model_type = modeltype, data_folder = data_folder)

data, label, params = load_and_filter_data(data_paths, params)

save_name = os.path.join(save_folder, f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}.h5')

finetune_eval_mode = False
if modeltype == 'Finetune':
    finetune_eval_mode = True
    params['modelpath'] = save_name.replace('Finetune','Orig') # the pre-trained model to be fine-tuned on
save_name = train_models(data, label, save_name, params)

#evaluation
model_path = save_name

eval_data_paths = generate_paths(subj_id, task, nclass, session_num, model_type = 'Finetune', data_folder = data_folder, finetune_eval_mode=finetune_eval_mode)
acc = evaluate_model(model_path, eval_data_paths, params)

# Define the metrics data
metrics_data = {
    'subject_id': [f'S{subj_id:02}'],
    'session_number': [session_num],
    'nclass': [nclass],
    'task': [task],
    'modeltype': [modeltype],
    'accuracy': [acc]
}

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_data)

# File path for the CSV
csv_file_path = 'metrics.csv'

# Check if the file exists
if os.path.exists(csv_file_path):
    # If it exists, append the new data
    existing_df = pd.read_csv(csv_file_path)
    updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
else:
    # If it doesn't exist, create a new file
    updated_df = metrics_df

# Save the updated DataFrame to the CSV file
updated_df.to_csv(csv_file_path, index=False)