import pandas as pd
import numpy as np
import config

import torch
import torch.nn as nn
import config
from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
)

def check_emotion_ids(emotion, tokenizer):
    return {'input_ids':tokenizer.convert_tokens_to_ids(emotion), 'token_type_ids':0, 'attention_mask':1}



class Dataset:
    def __init__(self, emotion, cause_utterance, cause_span, utterance, context):
        self.emotion = emotion
        self.cause_utterance = cause_utterance
        self.cause_span = cause_span
        self.utterance = utterance
        self.context = context
        self.tokenizer = config.TOKENIZER
        self.max_len = {
            'max_alen': config.MAX_ALENGTH,
            'max_qlen': config.MAX_QLENGTH,
            'max_clen': config.MAX_CLENGTH,
            'max_ulen': config.MAX_ULENGTH,
        }
        
    def __len__(self):
        return len(self.utterance)
    
    def encode(self, text1, text2, add_special_tokens, max_len, pad_to_max_len, truncation):
        return self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=add_special_tokens,
            max_length=max_len,
            padding=pad_to_max_len,
            truncation=truncation,
            return_offsets_mapping=True
        )


    def print_encoded(self, s, t):
        print(t)
        print(s['input_ids'])
        print(s['token_type_ids'])
        print(s['attention_mask'])
        print()


    def process_dataset(self, utterance, emotion, cause_span, cause_utterance, context, tokenizer, max_len):
        ### Encode the constant part
        S1 = 'The target utterance is '
        S2 = 'The evidence utterance is '
        S3 = "What is the causal span from context that is relevant to the target utterance's emotion "
        s1_encode = self.encode(S1, None, False, 32, False, True)
        s2_encode = self.encode(S2, None, False, 32, False, True)
        s3_encode = self.encode(S3, None, False, 32, False, True)
        
        '''
        self.print_encoded(s1_encode, self.tokenizer.tokenize(S1))
        self.print_encoded(s2_encode, self.tokenizer.tokenize(S2))
        self.print_encoded(s3_encode, self.tokenizer.tokenize(S3))
        '''

        start_idx = None
        end_idx = None

        # make the selected text characters as 1 and rest as 0
        char_targets = [0]*len(context)

        if start_idx != None and end_idx != None:
            for ch in range(start_idx, end_idx):
                char_targets[ch] = 1


        # emotion_encode = self.encode(emotion, None, False, 128, False, True)
        context_encode = self.encode(context, None, True, self.max_len['max_clen'], True, True)

        # -2 since we need to add [cls] and [sep] token in the utterance input ids
        utterance_encode = self.encode(utterance, None, False, self.max_len['max_ulen']-2, True, True)
        if cause_utterance is np.nan:
            cause_utterance_encode = self.encode("Nothing", None, False, self.max_len['max_ulen'], True, True)
        else:
            cause_utterance_encode = self.encode(cause_utterance, None, False, self.max_len['max_ulen'], True, True)

        
        overall_input_ids = []
        overall_input_ids.extend(context_encode['input_ids'])
        overall_input_ids.extend(s1_encode['input_ids'])
        overall_input_ids.extend(utterance_encode['input_ids'])
        overall_input_ids.extend(s2_encode['input_ids'])
        overall_input_ids.extend(cause_utterance_encode['input_ids'])
        overall_input_ids.extend(s3_encode['input_ids'])


        # <[CLS], len(context), [SEP]> <S1, utterance, S2, cause_utterance, S3> 
        # emotions are added in the forward function
        overall_token_type_ids = [0] * (len(context_encode['input_ids'])) + [1] * (len(overall_input_ids) - len(context_encode['input_ids'])) 

        overall_attention_mask = [1] * len(overall_input_ids)
        
        # emotion idx will be the index where the emotion needs to be inserted
        emotion_idx = len(overall_input_ids)

        padding_len = self.max_len['max_qlen'] - len(overall_input_ids)
        # print("Padding Length: ", padding_len)

        if padding_len > 0:
            overall_input_ids = overall_input_ids + ( [0]*padding_len )
            overall_token_type_ids = overall_token_type_ids + ( [1]*padding_len )
            overall_attention_mask = overall_attention_mask + ( [0]*padding_len )

        # print(overall_input_ids)

        utterance_input_ids = [101] + utterance_encode['input_ids'] + [102] 
        utterance_mask = [1]*len(utterance_input_ids)
        utterance_token_type_ids=[0]*len(utterance_input_ids)

        padding_ulen = self.max_len['max_ulen'] - len(utterance_input_ids)
        # print("padding ULength:", padding_ulen)

        if padding_ulen > 0:
            utterance_input_ids = utterance_input_ids + ([0]*padding_ulen)
            utterance_mask = utterance_mask + ([0]*padding_ulen)
            utterance_token_type_ids = utterance_token_type_ids + ([0]*padding_ulen)

        #### Creating dataset for cause span predictor
        start_idx = None
        end_idx = None  

        if cause_utterance is np.nan:
            return {
            'input_ids': overall_input_ids,
            'attention_mask': overall_attention_mask,
            'token_type_ids': overall_token_type_ids,
            'targets_start': -1,
            'targets_end': -1,
            'offset_start': 0,
            'offset_end': 0,
            'utterance_input_ids': utterance_input_ids,
            'utterance_mask': utterance_mask,
            'utterance_token_type_ids': utterance_token_type_ids,
            'emotion_idx': emotion_idx,
        }

        # if cause_utterance is not np.nan, then

        # look for the substring of cause span in the context (character level)
        for ind in (i for i, e in enumerate(context) if e == cause_span[0]):
            if context[ind: ind + len(cause_span)] == cause_span:
                start_idx = ind
                end_idx = ind + len(cause_span)
                break 

        # print(context)
        # print(cause_span)
        # print(start_idx, end_idx)
        # print()

        # make the selected text characters as 1 and rest as 0
        char_targets = [0]*len(context)

        if start_idx != None and end_idx != None:
            for ch in range(start_idx, end_idx):
                char_targets[ch] = 1
                

        # this section of code needs to be checked
        
        # offset_mapping tell us the starting and end index of a token at character level
        # 1 to -1 because we need to skip [CLS] and [SEP]
        context_offset_mapping = context_encode.offset_mapping[1:-1]

        # print(utterance)
        print(context_offset_mapping)
        print(char_targets)
        print(self.tokenizer.tokenize(context))
        print(len(self.tokenizer.tokenize(context)))
        # print(context)
        # print(cause_span)
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(context_offset_mapping):
            print(char_targets[offset1: offset2], end=' ')
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)
        
        print(cause_span) 
        print(target_idx)

        targets_start = target_idx[0]
        targets_end = target_idx[-1]
    
        # print(target_idx)
        # print(targets_start)
        # print(targets_end)


        # length = {
        #     'input_ids': type(overall_input_ids),
        #     'attention_mask': type(overall_attention_mask),
        #     'token_type_ids': type(overall_token_type_ids),
        #     'targets_start': type(targets_start),
        #     'targets_end': type(targets_end),
        #     'offset_start': 0,
        #     'offset_end': 0,
        #     'utterance_input_ids': len(utterance_input_ids),
        #     'utterance_mask': len(utterance_mask),
        #     'utterance_token_type_ids': len(utterance_token_type_ids),
        # }
        
        # print(context)
        # print("*"*50)
        # print(cause_utterance)
        # print("*"*50)
        # print(utterance)
        # print("*"*50)
        # print(length)
        # print()

        return {
            'input_ids': overall_input_ids,
            'attention_mask': overall_attention_mask,
            'token_type_ids': overall_token_type_ids,
            'targets_start': 0,
            'targets_end': 0,
            'offset_start': 0,
            'offset_end': 0,
            'utterance_input_ids': utterance_input_ids,
            'utterance_mask': utterance_mask,
            'utterance_token_type_ids': utterance_token_type_ids,
            'emotion_idx': emotion_idx,
        }


    def __getitem__(self, idx):
        
        context = ' '.join(self.context[idx].split()) # this just removes all the greater than single space whitespaces
        emotion = self.emotion[idx]
        utterance = ' '.join(self.utterance[idx].split())
        # print(utterance)
        if self.cause_utterance[idx] is np.nan:
            # print(self.cause_utterance[idx])
            cause_utterance = self.cause_utterance[idx]
            cause_span = self.cause_span[idx]

        else:
            cause_utterance = ' '.join(self.cause_utterance[idx].split())
            cause_span = ' '.join(self.cause_span[idx].split())
    
        inputs = self.process_data(utterance, emotion, cause_span, cause_utterance, context, self.tokenizer, self.max_len)
        
        length = {
            'input_ids': (inputs['input_ids']),
            'attention_mask': (inputs['attention_mask']),
            'token_type_ids': (inputs['token_type_ids']),
            'targets_start': (inputs['targets_start']),
            'targets_end': (inputs['targets_end']),
            'offset_start': 0,
            'offset_end': 0,
            'utterance_input_ids': (inputs['utterance_input_ids']),
            'utterance_mask': (inputs['utterance_mask']),
            'utterance_token_type_ids': (inputs['utterance_token_type_ids']),
            'offset_start': (inputs['offset_start']),
            'offset_end': (inputs['offset_end']),
            'emotion': (emotion),
            'emotion_idx': (inputs['emotion_idx'])
        }

        # print(length)
        # print()

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'targets_start': torch.tensor(inputs['targets_start'], dtype=torch.long), 
            'targets_end': torch.tensor(inputs['targets_end'], dtype=torch.long),
            'utterance_input_ids': torch.tensor(inputs['utterance_input_ids'], dtype=torch.long),
            'utterance_mask': torch.tensor(inputs['utterance_mask'], dtype=torch.long),
            'utterance_token_type_ids': torch.tensor(inputs['utterance_token_type_ids'], dtype=torch.long),
            'cause_span': cause_span,
            'offset_start': torch.tensor(inputs['offset_start'], dtype=torch.long),
            'offset_end': torch.tensor(inputs['offset_end'], dtype=torch.long),
            'emotion': torch.tensor(int(emotion), dtype=torch.float),
            'emotion_idx': torch.tensor(inputs['emotion_idx'], dtype=torch.long),
        }

if __name__ == '__main__':
    df_train = pd.read_csv(config.TRAIN_DATASET)
    df_valid = pd.read_csv(config.VALID_DATASET)

    df_train.emotion = df_train.emotion.apply(lambda x: config.emotion_mapping[x])
    df_valid.emotion = df_valid.emotion.apply(lambda x: config.emotion_mapping[x])

    # print(dfx_valid.head())
    
    train_dataset = Dataset(
        emotion = df_train.emotion.values,
        cause_utterance = df_train.cause_utterance.values,
        cause_span = df_train.cause_span.values,
        utterance = df_train.utterance.values,
        context = df_train.history.values,
    )

    print(train_dataset[390])