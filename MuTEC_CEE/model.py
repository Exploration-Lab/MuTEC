import torch
import torch.nn as nn
import config
from dataset import *
from utils import *
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import  tqdm 
from transformers import (
    AutoModel, 
    AutoConfig
)
import numpy as np
from sklearn import metrics
import os
import torch.nn.functional as F
import numpy as np
import wandb

class ClsModel(nn.Module):
    def __init__(self, model_name, model_id, one_layer=True, fc_dim=512, n_outputs=2,
    n_em_outputs=6, dropout_prob=0.1):
        super(ClsModel, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_id, output_hidden_states=True)
        self.encoder = AutoModel.from_pretrained(model_id, config=self.config)
        
        n_hidden = self.config.hidden_size//2
        self.dropout1 = nn.Dropout(self.config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.e_linear = nn.Linear(n_hidden*2, n_em_outputs)
        self.c_linear = nn.Linear(n_hidden*2, n_outputs)
        self.out_linear = nn.Linear(4*n_hidden + self.config.hidden_size, n_outputs)
        self.n_outputs = n_outputs
        self.linear = nn.Linear(768, 2)
        
        self.e_bilstm = nn.LSTM(self.config.hidden_size, n_hidden, batch_first=True, bidirectional=True)
        self.c_bilstm = nn.LSTM(self.config.hidden_size, n_hidden, batch_first=True, bidirectional=True)
        
        if not one_layer:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.config.hidden_size, out_features=fc_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
                nn.Linear(in_features= fc_dim, out_features=n_outputs)
            )
        else:
            self.classifier = nn.Sequential(nn.Linear(in_features=self.config.hidden_size, out_features=n_outputs))


    def get_embeddings(self, input_ids, attention_mask, token_type_ids):
        if self.model_name in ['roberta']:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        # out keys: ['last_hidden_state', 'pooler_output', 'hidden_states']
        seq_out, pool_out, hidden_out = out[0], out[1], out[2]

        # pool the last 4 [CLS] tokens hidden state
        h12 = hidden_out[-1][:, 0].reshape((-1, 1, self.config.hidden_size))
        h11 = hidden_out[-2][:, 0].reshape((-1, 1, self.config.hidden_size))
        h10 = hidden_out[-3][:, 0].reshape((-1, 1, self.config.hidden_size))
        h9  = hidden_out[-4][:, 0].reshape((-1, 1, self.config.hidden_size))
        all_hidden = torch.cat([h9, h10, h11, h12], dim=1)
        mean_pool = torch.mean(all_hidden, dim=1)
        return seq_out, mean_pool

    def get_emotion_embed(self, x):
      x_context, hidden_states = self.e_bilstm(x)
      x = self.dropout2(x_context)
      # l = x[:, -1, :] # for last time step
      l = torch.mean(x, dim=1)
      x = self.e_linear(l)

      return x_context, x, l
    
    def get_cause_embed(self, x):
      x_context, hidden_states = self.c_bilstm(x)
      x = self.dropout2(x_context)
      # l = x[:, -1, :] # for last time step
      l = torch.mean(x, dim=1)
      x = self.c_linear(l)       
      return x_context, x, l


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, ut_len=None, ui_len=None):
        s_out, pool_out = self.get_embeddings(input_ids, attention_mask, token_type_ids)
        pool, ut, ui, c = s_out.split([1, ut_len[0], ui_len[0], s_out.size(1)-ut_len[0]-ui_len[0]-1], dim=1)
        
        ut_embed, _, e = self.get_emotion_embed(ut)
        e_logits = self.e_linear(e)
        
        ui = torch.cat([ut_embed, ui], dim=1)
        _, _, c = self.get_cause_embed(ui)
        x = torch.cat([e, c, pool_out], dim=1)
        f = self.out_linear(x)
        
        return e_logits, f

class EntailModel:
    def __init__(self, path=None, device=None, model_name=None, model_id=None, out_weights=None, em_weights=None):
        self.model=ClsModel(model_name, model_id)
        self.path=path
        self.device=device
        self.model_name = model_name
        self.model_id = model_id
        self.out_weights = out_weights
        self.em_weights = em_weights
        self.example_cnt = 0
        self.batch_cnt = 0
        self.eval_example_cnt = 0
        self.eval_batch_cnt = 0
    
    def out_loss_fn(self, outputs, targets):
        weights = torch.tensor(self.out_weights, dtype=torch.float)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)
        return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets.long())
      
    def em_loss_fn(self, outputs, targets):
        weights = torch.tensor(self.em_weights, dtype=torch.float)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)
        return nn.CrossEntropyLoss(weight=class_weights)(outputs, targets.long())
      
    def train_model(self, df_train, df_valid=None):
        train_dataset = Dataset(
            Ut = df_train.Ut.values,
            Ui = df_train.Ui.values,
            context = df_train.context.values if 'context' in df_train.columns else None,
            labels = df_train.labels.values,
            em_labels = list(config.emotion_mapping[x] for x in df_train.emotion.values),
            model_name = self.model_name,
            model_id = self.model_id, 
        )

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, 
            sampler = torch.utils.data.RandomSampler(train_dataset),
            batch_size = config.TRAIN_BATCH_SIZE,
            num_workers=0
        )

        valid_dataset = Dataset(
            Ut = df_valid.Ut.values,
            Ui = df_valid.Ui.values,
            context = df_valid.context.values if 'context' in df_valid.columns else None,
            labels = df_valid.labels.values,
            em_labels = list(config.emotion_mapping[x] for x in df_valid.emotion.values),
            model_name = self.model_name,
            model_id = self.model_id, 
        )

        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            sampler = torch.utils.data.SequentialSampler(valid_dataset),
            batch_size = config.VALID_BATCH_SIZE,
            num_workers=0
        )
        
        model = self.model
        model.to(self.device)

        param_optimizer = list(model.named_parameters())
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

        num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCH)

        optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=num_train_steps
        )

        best_accuracy = -1
        best_f1 = -1
        train_loss_list = []
        eval_loss_list = []
        if config.WANDB:
            wandb.watch(model, self.em_loss_fn, log='all', log_freq=100)

        for epoch in range(config.EPOCH):
            train_loss = self.train_fn(train_data_loader, model, optimizer, self.device, scheduler)
            outputs, targets, em_outputs, em_targets, eval_loss = self.eval_fn(valid_data_loader, model, self.device)
            
            train_loss_list.append(train_loss)
            eval_loss_list.append(eval_loss)

            accuracy = metrics.accuracy_score(targets, outputs)
            em_accuracy = metrics.accuracy_score(em_targets, em_outputs)
            f1 = metrics.f1_score(targets, outputs)
            em_f1 = metrics.f1_score(em_targets, em_outputs, average='macro')

            if config.WANDB:
                wandb.log({"f1": f1, "em_f1": em_f1, "accuracy": accuracy, "em_accuracy": em_accuracy, \
                'train_loss_avg': train_loss, 'eval_loss_avg': eval_loss})
                # wandb.save('model.onnx')

            print(f"Epoch: {epoch}")
            print(f"Label Accuracy score= {accuracy}")
            print(f"Label F1 score= {f1}")
            print(f"Emotion Accuracy score= {em_accuracy}")
            print(f"Emotion F1 score= {em_f1}")

            if f1 > best_f1:
                save_path='epoch{}.pth'.format(epoch)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(self.path, save_path))
                best_accuracy = accuracy
                best_f1 = f1

    def train_fn(self, data_loader, model, optimizer, device, scheduler):
        model.train()
        train_loss = AverageMeter()

        for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc="Training")):
            input_ids = d['input_ids'].to(device, dtype=torch.long)
            attention_mask = d['attention_mask'].to(device, dtype=torch.long)
            ut_len = d['ut_len'].to(device, dtype=torch.long)
            ui_len = d['ui_len'].to(device, dtype=torch.long)
            targets = d['targets'].to(device, dtype=torch.float)
            em_targets = d['em_targets'].to(device, dtype=torch.float)
            
            model.zero_grad()
            
            if self.model_name in ['roberta']:
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    ut_len=ut_len,
                    ui_len=ui_len,
                )
            else:
                token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    ut_len=ut_len,
                    ui_len=ui_len,
                )

            ut_preds, final_preds = outputs
            loss = config.BETA * self.em_loss_fn(ut_preds, em_targets) + self.out_loss_fn(final_preds, targets)

            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            self.example_cnt += input_ids.size(0)
            self.batch_cnt += 1

            if config.WANDB:
                if ((self.batch_cnt + 1) % 100) == 0:
                    wandb.log({'train_loss': loss}, step=self.example_cnt)
                    # plot_grad_flow(model.named_parameters(), './viz_gradients', self.example_cnt)

            train_loss.update(loss.item(), targets.size(0))
        
        return train_loss.avg

    def eval_fn(self, data_loader, model, device):
        model.eval()
        eval_loss = AverageMeter()

        fin_targets = []
        fin_outputs = []
        e_outputs = []
        e_targets = []
        with torch.no_grad():
            for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc="Evaluation")):
                input_ids = d['input_ids'].to(device, dtype=torch.long)
                attention_mask = d['attention_mask'].to(device, dtype=torch.long)
                ut_len = d['ut_len'].to(device, dtype=torch.long)
                ui_len = d['ui_len'].to(device, dtype=torch.long)      
                targets = d['targets'].to(device, dtype=torch.float)
                em_targets = d['em_targets'].to(device, dtype=torch.float)
                
                if self.model_name in ['roberta']:
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        ut_len=ut_len,
                        ui_len=ui_len,
                    )
                else:
                    token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
                    outputs = model(
                        input_ids=input_ids, 
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        ut_len=ut_len,
                        ui_len=ui_len,
                    )
                ut_preds, final_preds = outputs
                
                if 'iemocap' in config.TEST_DATASET:
                    ut_preds = ut_preds[:, [0, 2, 3, 6]]
                    dd_to_iemo = {
                        0: 0,
                        2: 1,
                        3: 2,
                        6: 3 
                    }
                    em_targets_cpu = em_targets.cpu().detach().numpy().tolist()
                    em_targets_cpu = [dd_to_iemo[x] for x in em_targets_cpu]
                    e_targets.extend(em_targets_cpu)
                    e_outputs.extend(torch.argmax(ut_preds, axis=1).cpu().detach().numpy().tolist())
                else:
                    e_targets.extend(em_targets.cpu().detach().numpy().tolist())
                    e_outputs.extend(torch.argmax(ut_preds, axis=1).cpu().detach().numpy().tolist())

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.argmax(final_preds, axis=1).cpu().detach().numpy().tolist())
            
                self.eval_example_cnt += input_ids.size(0)
                self.eval_batch_cnt += 1


        return fin_outputs, fin_targets, e_outputs, e_targets, eval_loss.avg

    def eval_model(self, df_test):
        test_dataset = Dataset(
            Ut = df_test.Ut.values,
            Ui = df_test.Ui.values,
            context = df_test.context.values if 'context' in df_test.columns else None,
            labels = df_test.labels.values,
            em_labels = list(config.emotion_mapping[x] for x in df_test.emotion.values),
            model_name = self.model_name,
            model_id = self.model_id,   
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, 
            sampler = torch.utils.data.SequentialSampler(test_dataset),
            batch_size = config.VALID_BATCH_SIZE,
            num_workers=0
        )

        ckpt = torch.load(self.path)
        self.model.load_state_dict(ckpt['model_state_dict'])
        model = self.model.to(self.device)
        outputs, targets, em_outputs, em_targets, eval_loss = self.eval_fn(test_data_loader, model, self.device)
        
        return outputs, targets, em_outputs, em_targets
