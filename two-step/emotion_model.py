import torch.nn as nn
import torch
import torch.nn.functional as F
import dataset 
import os

from tqdm import tqdm
from transformers import (
    AutoModel,
)
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import config
from sklearn import metrics
from utils import *
import wandb
import config

# seed_torch()

class EmotionModel(nn.Module): 
    def __init__(self, one_layer=True, h_dim=512):
        super(EmotionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(config.emotion_model_name, config=config.EMOTION_CONFIG, cache_dir=config.TRANSFORMER_CACHE)
        if one_layer:
          self.output = nn.Linear(config.EMOTION_CONFIG.hidden_size, config.n_classes)
        else:
          self.output = nn.Sequential(nn.Linear(in_features=config.EMOTION_CONFIG.hidden_size, out_features=h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=0.1),
                    nn.Linear(in_features=h_dim, out_features=config.n_classes))


    def forward(self, utterance_input_ids, utterance_token_type_ids, utterance_mask):
        output = self.bert(
            input_ids=utterance_input_ids,
            token_type_ids=utterance_token_type_ids,
            attention_mask=utterance_mask,
        ) 
        out = self.output(output[1])
        return out

class EmotionPredictor:
    def __init__(self, path, weights=None, device='cuda', model_args=None):
        self.model = EmotionModel()
        self.path = path
        self.weights = weights
        self.device = device
        self.model_args = model_args

    def _get_inputs_dict(self, batch, is_training=False):
        if is_training:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "utterance_input_ids": batch[5],
                "utterance_token_type_ids": batch[6],
                "utterance_mask": batch[7],
                "emotion": batch[8],
                "emotion_idx": batch[9],
            }
        else:
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "utterance_input_ids": batch[3],
                "utterance_token_type_ids": batch[4],
                "utterance_mask": batch[5],
                "emotion": batch[6],
                "emotion_idx": batch[7],
            }

        # if config.model_type in ["xlm", "roberta", "distilbert", "camembert", "electra", "xlmroberta", "bart"]:
        #     del inputs["token_type_ids"]

        # if self.args.model_type in ["xlnet", "xlm"]:
        #     inputs.update({"cls_index": batch[5], "p_mask": batch[6]})

        return inputs

    # weighted cross entropy
    def loss_fn(self, outputs, targets):
        weights = torch.tensor(self.weights, dtype=torch.float)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)

        return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets.long())


    def train_fn(self, data_loader, model, optimizer, device, scheduler):
        model.train()
        train_loss = AverageMeter()

        for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc="Emotion Training")):
            inputs = self._get_inputs_dict(d, is_training=True)

            utterance_input_ids = inputs['utterance_input_ids']
            utterance_token_type_ids = inputs['utterance_token_type_ids']
            utterance_mask = inputs['utterance_mask']
            target = inputs['emotion']
            
            utterance_input_ids = utterance_input_ids.to(device, dtype=torch.long)
            utterance_token_type_ids = utterance_token_type_ids.to(device, dtype=torch.long)
            utterance_mask = utterance_mask.to(device, dtype=torch.long)
            target = target.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(
                utterance_input_ids=utterance_input_ids, 
                utterance_token_type_ids=utterance_token_type_ids, 
                utterance_mask=utterance_mask,
            )   

            loss = self.loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.update(loss.item(), target.size(0))
        
        return train_loss.avg

    def eval_fn(self, data_loader, model, device):
        model.eval()
        eval_loss = AverageMeter()

        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc='Emotion Evaluation')):
                inputs = self._get_inputs_dict(d, is_training=False)

                utterance_input_ids = inputs['utterance_input_ids']
                utterance_token_type_ids = inputs['utterance_token_type_ids']
                utterance_mask = inputs['utterance_mask']
                target = inputs['emotion']
            
                utterance_input_ids = utterance_input_ids.to(device, dtype=torch.long)
                utterance_token_type_ids = utterance_token_type_ids.to(device, dtype=torch.long)
                utterance_mask = utterance_mask.to(device, dtype=torch.long)
                target = target.to(device, dtype=torch.float)

                outputs = model(
                    utterance_input_ids=utterance_input_ids, 
                    utterance_token_type_ids=utterance_token_type_ids, 
                    utterance_mask=utterance_mask,
                )

                loss = self.loss_fn(outputs, target)
                eval_loss.update(loss.item(), target.size(0))

                fin_targets.extend(target.cpu().detach().numpy().tolist())
                fin_outputs.extend(F.softmax(outputs, dim=1).cpu().detach().numpy().tolist())
        
        fin_outputs = np.argmax(np.array(fin_outputs), axis=1)
        accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
        f1 = metrics.f1_score(fin_targets, fin_outputs, average='macro')
        # print("Output")
        # print(fin_outputs)
        # print("#"*50)
        # print(fin_targets)

        print(f"Accuracy score= {accuracy}\nF1 Score= {f1}")
        return fin_outputs, accuracy, f1, eval_loss.avg

    def train(self, device, train_data_loader, valid_data_loader, train_len):
        self.model.to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay': 0.001
            },
            {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
            }
        ]

        num_train_steps = int(train_len/self.model_args.train_batch_size * self.model_args.epochs)

        optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=num_train_steps
        ) 
        # model = nn.DataParallel(model) # for multiple GPUs

        best_accuracy = 0
        best_f1 = -1
        best_valid_predictions = []
        train_loss_list = []
        eval_loss_list = []

        for epoch in range(self.model_args.epochs):
            train_loss = self.train_fn(train_data_loader, self.model, optimizer, device, scheduler)
            outputs, accuracy, f1, eval_loss = self.eval_fn(valid_data_loader, self.model, device)
            
            train_loss_list.append(train_loss)
            eval_loss_list.append(eval_loss)
            if config.WANDB:
                wandb.log({"em_f1": f1, "em_accuracy": accuracy, "em_train_loss": train_loss, "em_val_loss": eval_loss})

            if f1 > best_f1:
                save_path = 'emotion_epoch_{}.pth'.format(epoch)
                # save_path = 'emotion_epoch.pt'
                torch.save(self.model.state_dict(), os.path.join(self.path, save_path))
                best_accuracy = accuracy
                best_valid_predictions = outputs
                best_f1 = f1

        # plot_loss(train_loss_list, eval_loss_list, "EP")
        return best_valid_predictions

    def evaluate(self, device, test_data_loader):
        self.model.load_state_dict(torch.load(self.path))
        self.model = self.model.to(device)

        outputs, accuracy, f1, eval_loss = self.eval_fn(test_data_loader, self.model, device)
        return outputs

        
