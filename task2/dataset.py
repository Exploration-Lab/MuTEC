import config
import torch
from transformers import AutoTokenizer, AutoConfig

class Dataset:
  ut_token_len = []
  ui_token_len = []

  def __init__(self, Ut, Ui, context, labels, em_labels, model_name, model_id):
    self.Ut = Ut
    self.Ui = Ui
    self.context = context
    self.labels = labels
    self.em_labels = em_labels
    self.max_len = config.MAX_LENGTH
    self.config = AutoConfig.from_pretrained(model_id, cache_dir=config.TRANSFORMER_CACHE)
    self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=config.TRANSFORMER_CACHE, config=self.config)
    self.model_name = model_name

  def __len__(self):
    return len(self.Ut)
  
  def __getitem__(self, idx):
    Ut = self.Ut[idx]
    Ui = self.Ui[idx]
    if self.context is None:
      context = ''
    else:
      context = self.context[idx]
    sep = self.tokenizer.sep_token

    ut_len = len(self.tokenizer.tokenize(Ut))
    ui_len = len(self.tokenizer.tokenize(Ui))
  
    Dataset.ut_token_len.append(ut_len)
    Dataset.ui_token_len.append(ui_len)
    
    if self.model_name in ['roberta']:
      text = Ut + Ui + sep + sep + context
    else:
      text = Ut + Ui + sep + context
    
    inputs = self.tokenizer.encode_plus(
      text,
      None, 
      add_special_tokens=True,
      max_length=self.max_len,
      padding='max_length',
      truncation=True
    )

    equal_len = True
    if equal_len:
      inputs = {}
      ut_len = ui_len = 40
      unk = self.tokenizer.unk_token
      pad_id = self.tokenizer.pad_token_id
      cls_id = self.tokenizer.cls_token_id
      sep_id = self.tokenizer.sep_token_id

      ut_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(Ut) + [unk]*ut_len)[:ut_len]
      ui_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(Ui) + [unk]*ui_len)[:ui_len]

      if context is None:
        if self.model_name not in ['roberta']:
          ids = [cls_id] + ut_id + [sep_id] + ui_id + [sep_id]
          ttids = [0]*(len(ut_id) + 2) + [1]*(len(ui_id) + 1)
        else:
          ids = [cls_id] + ut_id + [sep_id] + [sep_id] + ui_id + [sep_id]
        attn_mask = [1]*len(ids)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
          ids = ids + [pad_id]*pad_len
          attn_mask = attn_mask + [0]*pad_len
        if pad_len < 0:
          ids = ids[:self.max_len-1] + [sep_id]
          attn_mask = attn_mask[:self.max_len]
          
      else:
        if self.model_name not in ['roberta']:
          c_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sep + context))
        else:
          c_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sep + sep + context))
        ids = [cls_id] + ut_id + ui_id + c_id + [sep_id]
        attn_mask = [1]*len(ids)
        if self.model_name not in ['roberta']:
          ttids = [0]*(len(ut_id + ui_id) + 2) + [1]*(len(c_id)-1)
          inputs['token_type_ids'] = ttids

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
          ids = ids + [pad_id]*pad_len
          attn_mask = attn_mask + [0]*pad_len
        if pad_len < 0:
          ids = ids[:self.max_len-1] + [sep_id]
          attn_mask = attn_mask[:self.max_len]

      inputs['input_ids'] = ids
      inputs['attention_mask'] = attn_mask

    # print([(k, len(x)) for k, x in inputs.items()])
    # print(inputs)
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    if self.model_name in ['roberta']:
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'ut_len': torch.tensor(ut_len, dtype=torch.long),
            'ui_len': torch.tensor(ui_len, dtype=torch.long),
            'targets': torch.tensor(int(self.labels[idx]), dtype=torch.float),
            'em_targets': torch.tensor(int(self.em_labels[idx]), dtype=torch.float),
        }
    else:
        token_type_ids = inputs['token_type_ids']
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'ut_len': torch.tensor(ut_len, dtype=torch.long),
            'ui_len': torch.tensor(ui_len, dtype=torch.long),
            'targets': torch.tensor(int(self.labels[idx]), dtype=torch.float),
            'em_targets': torch.tensor(int(self.em_labels[idx]), dtype=torch.float),
        }
        

