import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as  pd
from transformers import (
    BertModel,
)
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from question_answering_utils import *
import numpy as np
import config
from sklearn import metrics
from question_answering_utils import ( 
    get_best_indexes,
    get_final_text,
    compute_softmax,

)

from gensim.models import Word2Vec
import torch.nn as nn
import wandb
from sklearn import metrics
from utils_e2e import *
import config


class BeamStart(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.linear = nn.Linear(model_config.hidden_size, 1)
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)


    def forward(self, hidden_states, p_mask=None):
        # x = self.linear(hidden_states).squeeze(-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        x = torch.mean(torch.stack([
                self.linear(self.high_dropout(hidden_states))
                for _ in range(5)
            ], dim=0), dim=0).squeeze(-1)

        return x

class BeamEnd(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.linear_0 = nn.Linear(model_config.hidden_size*2, model_config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps if hasattr(model_config,"layer_norm_eps") else 1e-5)
        self.linear_1 = nn.Linear(model_config.hidden_size, 1)
        self.linear = nn.Linear(model_config.hidden_size*2, 1)
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)


    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        # start_positions = None
        if start_positions is not None:
            slen, hs = hidden_states.shape[-2:]
            st_pos = start_positions[:, None, None].expand(-1, -1, hs)  # shape (bs, 1, hs). same position in all 768 entries
            start_states = hidden_states.gather(-2, st_pos)  # shape (bs, 1, hs)
            start_states = start_states.expand(-1, slen, -1)  # shape (bs, slen, hs).

        x = self.linear_0(torch.cat([hidden_states, start_states], dim=-1)) # concatenates start_state with embed of start pos. in all 512 tokens with hidden states
        x = self.activation(x)

        # out = torch.cat([hidden_states, start_states], dim=-1)
        # # Multisample Dropout: https://arxiv.org/abs/1905.09788
        x = torch.mean(torch.stack([
                self.linear_1(self.high_dropout(x))
                for _ in range(5)
            ], dim=0), dim=0).squeeze(-1)

        # x = self.linear_0(torch.cat([hidden_states, start_states], dim=-1)) # concatenates start_state with embed of start pos. in all 512 tokens with hidden states
        # x = self.activation(x)
        # x = self.LayerNorm(x)
        # x = self.linear_1(x).squeeze(-1)
        
        return x

