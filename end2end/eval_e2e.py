import config
import pandas as pd
from e2e_model_v1 import *
# from e2e_modify import *
import argparse

from utils_e2e import *

from sklearn import metrics
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    BertConfig
)
import wandb
from utils_e2e import *
import os


def run(device):
    df_train = pd.read_csv(config.TRAIN_DATASET)
    df_valid = pd.read_csv(config.VALID_DATASET)
    df_test = pd.read_csv(config.TEST_DATASET)

    # df_train = df_train[:500]
    # df_valid = df_valid[:100]
    # df_test = df_test[:100]

    df_train = preprocess(df_train)
    df_valid = preprocess(df_valid)
    df_test = preprocess(df_test)

    df_train.emotion = df_train.emotion.apply(lambda x: config.emotion_mapping[x])
    df_valid.emotion = df_valid.emotion.apply(lambda x: config.emotion_mapping[x])
    df_test.emotion = df_test.emotion.apply(lambda x: config. emotion_mapping[x])

    em_weights = [0]*6
    em = df_train['emotion'].value_counts().reset_index().values.tolist()  
  
    for idx, (e, c) in enumerate(em):
      em_weights[e] = c
    neg_weight = get_neg_weights(df_train)

    print("Evaluate model on test set")
    best_model_path = config.PATH
    for i, filename in enumerate(os.listdir(best_model_path)):
      if filename.endswith(".pth"): 
        print(f"{filename}\t File:{i}")    
        model = RecModel(device=device, best_model=os.path.join(best_model_path, filename), em_weights=em_weights, neg_weights=neg_weight)

        print("***************** DEV SET ************************")

        results, text = model.eval_fn(df_valid)
        r, _ = evaluate_results(text)
        print(r)
        
        print("***************** TEST SET ************************")
        results, text = model.eval_fn(df_test)
        r, _ = evaluate_results(text)
        print(r)
        

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    args = parser.parse_args()
    print(args)

    device = "cuda:{}".format(str(args.cuda))
    seed_everything(seed=config.SEED)
    sweep_config = config.sweep_config
    
    if config.WANDB:
      sweep_id = wandb.sweep(
                  sweep_config, 
                  project='one_step', 
                  entity='ashwani345'
                )

      wandb.init(project='one_step', 
                entity='ashwani345',
                config={
                  'epochs': config.EPOCHS,
                  'train_batch_size': config.TRAIN_BATCH_SIZE, 
                  'valid_batch_size': config.VALID_BATCH_SIZE, 
                  'accumulation_steps': config.ACCUMULATION_STEPS,
                  'max_qlength': config.MAX_QLENGTH,
                  'max_alength': config.MAX_ALENGTH,
                  'max_clength': config.MAX_CLENGTH,
                  'max_ulength': config.MAX_ULENGTH,
                  'test_dataset': config.TEST_DATASET,
                })

    model_args = wandb.config
    os.environ["WANDB_API_KEY"] = "ebbd83c0708fcda75d8830954d38aec2241b7637"
    # wandb.agent(sweep_id, train)
    run(device)
