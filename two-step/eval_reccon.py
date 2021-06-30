import config
import pandas as pd
import dataset
from model import *
import argparse
import utils
from sklearn import metrics
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch
import wandb

def run(device, model_args):
    df_train = pd.read_csv(config.TRAIN_DATASET)
    df_valid = pd.read_csv(config.VALID_DATASET)
    df_test = pd.read_csv(config.TEST_DATASET)

    df_train = preprocess(df_train)
    df_valid = preprocess(df_valid)
    df_test = preprocess(df_test)

    # df_train = df_train[:100]
    # df_valid = df_valid[:100]
    # df_test = df_test[:1000]

    df_train.emotion = df_train.emotion.apply(lambda x: config.emotion_mapping[x])
    df_valid.emotion = df_valid.emotion.apply(lambda x: config.emotion_mapping[x])
    df_test.emotion = df_test.emotion.apply(lambda x: config. emotion_mapping[x])
    
    em_weights = [0]*7
    em = df_train['emotion'].value_counts().reset_index().values.tolist()  
  
    for idx, (e, c) in enumerate(em):
      em_weights[e] = c
    
    ep_best_model_path = 'ep_models/'
    cs_best_model_path = 'cs_models/'
    for filename1 in os.listdir(ep_best_model_path):
      if filename1.endswith(".pth"): 
        for filename2 in os.listdir(cs_best_model_path):
          if filename2.endswith(".pth"): 
            print(filename1, filename2)
            model = RecModel(device, os.path.join(ep_best_model_path, filename1), \
            os.path.join(cs_best_model_path, filename2), em_weights, model_args)
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
    seed_torch(config.SEED)
    model_args = {
      'epochs': 8,
      'train_batch_size': 14, 
      'valid_batch_size': 14, 
      'accumulation_steps': 4,
      'max_qlength': 512,
      'max_alength': 200,
      'max_clength': 512,
      'max_ulength': 512,
      'train_dataset': config.TRAIN_DATASET,
      'test_dataset': config.TEST_DATASET
    }

    if config.WANDB:
      wandb.init(project='two_step', 
                entity='ashwani345',
                config=model_args)

      model_args = wandb.config
      os.environ["WANDB_API_KEY"] = "ebbd83c0708fcda75d8830954d38aec2241b7637"
    else:
      model_args = utils.dotdict(model_args)

    run(device, model_args)
