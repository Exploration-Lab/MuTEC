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

import wandb
from utils_e2e import *

def run(device):
    df_train = pd.read_csv(config.TRAIN_DATASET)
    df_valid = pd.read_csv(config.VALID_DATASET)
    df_test = pd.read_csv(config.TEST_DATASET)

    # df_train = df_train[:100]
    # df_valid = df_valid[:50]
    # df_test = df_test[:50]

    df_train = preprocess(df_train)
    df_valid = preprocess(df_valid)
    df_test = preprocess(df_test)

    df_train.emotion = df_train.emotion.apply(lambda x: config.emotion_mapping[x])
    df_valid.emotion = df_valid.emotion.apply(lambda x: config.emotion_mapping[x])
    df_test.emotion = df_test.emotion.apply(lambda x: config. emotion_mapping[x])

    ###
    em_weights = [0]*6
    em = df_train['emotion'].value_counts().reset_index().values.tolist()  
  
    for idx, (e, c) in enumerate(em):
      em_weights[e] = c
    ####

    neg_weight = get_neg_weights(df_train)

    best_model_path = config.PATH

    model = RecModel(device=device, best_model=best_model_path, em_weights=em_weights, neg_weights=neg_weight)
    model.train_fn(df_train, df_valid)

    # print("Evaluate model on test set")
    # best_model_path = 'models/epoch.pth'
    # model = RecModel(device=device, best_model=best_model_path, em_weights=em_weights)
    # results, text = model.eval_fn(df_test)
    
    # print("Actual", actual)
    # print("Predicted", predicted)
    # accuracy = metrics.accuracy_score(actual, predicted)
    # print(f"Emotion Accuracy score= {accuracy}")
    # r = evaluate_results(text)
    # print(r)


if __name__ == '__main__':
    
    global args
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    args = parser.parse_args()
    print(args)

    device = "cuda:{}".format(str(args.cuda))
    print(device)
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
                  "beam_size": config.BEAM_SIZE,
                })

      model_args = wandb.config
    os.environ["WANDB_API_KEY"] = "ebbd83c0708fcda75d8830954d38aec2241b7637"
    # wandb.agent(sweep_id, train)
    run(device)
