import numpy as np, pandas as pd
import json, os, logging, pickle, argparse
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from model import EntailModel
import config 
from utils import *
import wandb
import os
import dataset

def run(device, model_name, model_id):
    df_train = pd.read_csv(config.TRAIN_DATASET)
    df_valid = pd.read_csv(config.VALID_DATASET)
    df_test = pd.read_csv(config.TEST_DATASET)

    df_train = preprocess(df_train)
    df_valid = preprocess(df_valid)
    df_test = preprocess(df_test)
    
    out_weights, em_weights = get_weights(df_train)

    print(df_train['emotion'].value_counts())
    print(df_valid['emotion'].value_counts())
    print(df_test['emotion'].value_counts())
    
    path = 'best_model/'
    filename = '.pth'

    model = EntailModel(path=os.path.join(path, filename), device=device, model_name=model_name, model_id=model_id, out_weights=out_weights, em_weights=em_weights)
    outputs, targets, em_outputs, em_targets = model.eval_model(df_test)
    
    if 'iemocap' in config.TEST_DATASET:
      r = str(classification_report(em_targets, em_outputs, digits=4, zero_division=0, target_names=['happiness', 'anger', 'sadness', 'excited'],  ))
    else:
      r = str(classification_report(em_targets, em_outputs, digits=4, zero_division=0))

    print(str(classification_report(targets, outputs, digits=4, zero_division=0)))


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    args = parser.parse_args()

    print(args)
    model_name = 'roberta'
    model_id = 'roberta-base'
    device = "cuda:{}".format(str(args.cuda))
    if config.WANDB:
      wandb.init(project='...', 
                entity='...',
                config={
                  'epochs':config.EPOCH,
                  'train_batch_size': config.TRAIN_BATCH_SIZE, 
                  'valid_batch_size': config.VALID_BATCH_SIZE, 
                  'accumulation_steps': config.ACCUMULATION_STEPS,
                  'max_length': config.MAX_LENGTH,
                  'lr': config.LEARNING_RATE,
                })
      
      model_args = wandb.config
      os.environ["WANDB_API_KEY"] = "..."

    run(device, model_name, model_id)


