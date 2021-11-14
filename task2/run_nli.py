import numpy as np, pandas as pd
import json, os, logging, pickle, argparse
from sklearn.metrics import classification_report, accuracy_score
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

    # df_train = df_train[:100]
    # df_valid = df_valid[:100]
    # df_test = df_test[:100]
    
    out_weights, em_weights = get_weights(df_train)
    
    print(out_weights, em_weights)
    print(df_train['emotion'].value_counts())
    print(df_valid['emotion'].value_counts())
    print(df_test['emotion'].value_counts())
    
    path = config.PATH
    model = EntailModel(path=path, device=device, model_name=model_name, model_id=model_id, out_weights=out_weights, em_weights=em_weights)
    model.train_model(df_train, df_valid=df_valid)

    '''
    model = EntailModel(path=path+'epoch.pth', device=device, model_name=model_name, model_id=model_id, out_weights=out_weights, em_weights=em_weights, model_args=model_args)
    outputs, targets, em_outputs, em_targets = model.eval_model(df_test)
    print("Emotion Accuracy: ", accuracy_score(em_targets, em_outputs))
    print(str(classification_report(em_targets, em_outputs, digits=3, zero_division=0)))
    r = str(classification_report(targets, outputs, digits=3, zero_division=0))
    print (r)
    
    
    plot_len(dataset.Dataset.ut_token_len, 'ut_length_dist')
    plot_len(dataset.Dataset.ui_token_len, 'ui_length_dist')
    '''

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
      wandb.init(project='reccon_sub2', 
                entity='ashwani345',
                config={
                  'epochs':config.EPOCH,
                  'train_batch_size': config.TRAIN_BATCH_SIZE, 
                  'valid_batch_size': config.VALID_BATCH_SIZE, 
                  'accumulation_steps': config.ACCUMULATION_STEPS,
                  'max_length': config.MAX_LENGTH,
                  'lr': config.LEARNING_RATE,
                  'dataset': config.TEST_DATASET,
                })

      model_args = wandb.config
      os.environ["WANDB_API_KEY"] = "ebbd83c0708fcda75d8830954d38aec2241b7637"

    run(device, model_name, model_id)


