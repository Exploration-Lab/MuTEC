import torch.nn as nn
import torch
import torch.nn.functional as F
import dataset 
from torch.utils.data import (
    DataLoader, 
    Dataset, 
    RandomSampler, 
    SequentialSampler, 
    TensorDataset
)

from utils import *
from question_answering_utils import (
    LazyQuestionAnsweringDataset,
    RawResult,
    RawResultExtended,
    build_examples,
    convert_examples_to_features,
    get_best_predictions,
    get_best_predictions_extended,
    get_examples,
    squad_convert_examples_to_features,
    to_list,
    write_predictions,
    write_predictions_extended,
    load_and_cache_examples
)

from tqdm import tqdm
from transformers import (
    BertForQuestionAnswering,
    BertModel,
    BertPreTrainedModel,
    BertForSequenceClassification,
)
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import numpy as np
import config

from emotion_model import EmotionPredictor
from cause_span_model import CauseSpanPredictor

# seed_torch()

class RecModel(nn.Module):
    def __init__(self, device, ep_best_model_path=None, cs_best_model_path=None, weights=None, model_args=None):
        super(RecModel, self).__init__()
        self.emotion_predictor = EmotionPredictor(ep_best_model_path, weights, device, model_args)
        self.cause_span_predictor = CauseSpanPredictor(cs_best_model_path, model_args)
        self.model_args = model_args
        self.device = device

        # print("............Inside init RecModel.........")

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False):
        """
        Converts a list of examples to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        tokenizer = config.CAUSE_TOKENIZER

        # examples in the input is a dataframe
        # get example returns a list of examples of SquadExample object
        examples = get_examples(examples, is_training=not evaluate)

        # if evaluate is true then mode is 'dev' else mode is 'train'
        mode = "dev" if evaluate else "train"

        # convert squad examples to features
        # examples are SquadExample generator
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=self.model_args.max_clength,
            doc_stride=256,
            max_query_length=self.model_args.max_qlength,
            is_training=not evaluate,
            model_args=self.model_args,
            # tqdm_enabled=not args.silent,
            # threads=args.process_count,
            # args=args,
        )
        
        if output_examples:
            return dataset, examples, features
        return dataset

    def modify_dataset(self, dataset, predictions=None, examples=None, features=None, is_training=False):
        # print("*******Inside Modify dataset*************")

        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_cls_index = []
        all_p_mask= []
        all_is_impossible = []
        all_utterance_input_ids = []
        all_utterance_token_type_ids = []
        all_utterance_mask = []
        all_emotion = []
        all_emotion_idx = []

        if is_training:
            all_start_positions = []
            all_end_positions = []

            for idx, d in enumerate(tqdm(dataset, total=len(dataset), desc="Modifying dataset")):
                ds = list(dataset[idx])
                emotion_idx = ds[9]
                emotion = ds[8]
                emotion = idx_to_emotion(emotion.tolist())
                emotion_iids = config.CAUSE_TOKENIZER.convert_tokens_to_ids([emotion])
                # 0: all input ids, just need to change input ids only
                ds[0][emotion_idx] = emotion_iids[0]
                all_input_ids.append(ds[0].tolist())
                all_attention_masks.append(ds[1].tolist())
                all_token_type_ids.append(ds[2].tolist())
                all_start_positions.append(ds[3].tolist())
                all_end_positions.append(ds[4].tolist())
                all_utterance_input_ids.append(ds[5].tolist())
                all_utterance_token_type_ids.append(ds[6].tolist())
                all_utterance_mask.append(ds[7].tolist())
                all_emotion.append(ds[8].tolist())
                all_emotion_idx.append(ds[9].tolist())
                all_cls_index.append(ds[10].tolist())
                all_p_mask.append(ds[11].tolist())
                all_is_impossible.append(ds[12].tolist())

                # ds = tuple(ds)
                # print(ds[1])
                # print(ds[2])

            # print(all_input_ids)

            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
            all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
            all_cls_index = torch.tensor(all_cls_index, dtype=torch.long)
            all_p_mask = torch.tensor(all_p_mask, dtype=torch.float)
            all_is_impossible = torch.tensor(all_is_impossible, dtype=torch.float)
            all_start_positions = torch.tensor(all_start_positions, dtype=torch.long)
            all_end_positions = torch.tensor(all_end_positions, dtype=torch.long)
        
            all_utterance_input_ids = torch.tensor(all_utterance_input_ids, dtype=torch.long)
            all_utterance_token_type_ids = torch.tensor(all_utterance_token_type_ids, dtype=torch.long)
            all_utterance_mask = torch.tensor(all_utterance_mask, dtype=torch.long)
            all_emotion = torch.tensor(all_emotion, dtype=torch.long)
            all_emotion_idx = torch.tensor(all_emotion_idx, dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_utterance_input_ids,
                all_utterance_token_type_ids,
                all_utterance_mask,
                all_emotion,
                all_emotion_idx,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )
    
        else:
            # valid_dataset, examples, features all have same length
            for idx, d in tqdm(enumerate(dataset), total=len(dataset), desc="Modifying dataset"):
                # modify valid_dataset
                ds = list(dataset[idx])
                emotion_idx = ds[7]
                emotion = torch.tensor(predictions[idx])
                emotion = idx_to_emotion(emotion.tolist())

                emotion_iids = config.CAUSE_TOKENIZER.convert_tokens_to_ids([emotion])
                # 0: all input ids, just need to change input ids only
                ds[0][emotion_idx] = emotion_iids[0]
                all_input_ids.append(ds[0].tolist())
                all_attention_masks.append(ds[1].tolist())
                all_token_type_ids.append(ds[2].tolist())
                # all_start_positions.append(ds[3].tolist())
                # all_end_positions.append(ds[4].tolist())
                all_utterance_input_ids.append(ds[3].tolist())
                all_utterance_token_type_ids.append(ds[4].tolist())
                all_utterance_mask.append(ds[5].tolist())
                all_emotion.append(ds[6].tolist())
                all_emotion_idx.append(ds[7].tolist())
                all_cls_index.append(ds[9].tolist())
                all_p_mask.append(ds[10].tolist())

                # modify examples
                e_list = examples[idx].question_text.split()
                l = len(e_list)
                e_list.append(emotion)
                examples[idx].question_text = ' '.join(e_list)
            
                # modify features
                features[idx].input_ids[emotion_idx] = emotion_iids[0]

            all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)
            all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
            all_cls_index = torch.tensor(all_cls_index, dtype=torch.long)
            all_p_mask = torch.tensor(all_p_mask, dtype=torch.float)
            
            all_utterance_input_ids = torch.tensor(all_utterance_input_ids, dtype=torch.long)
            all_utterance_token_type_ids = torch.tensor(all_utterance_token_type_ids, dtype=torch.long)
            all_utterance_mask = torch.tensor(all_utterance_mask, dtype=torch.long)
            all_emotion = torch.tensor(all_emotion, dtype=torch.long)
            all_emotion_idx = torch.tensor(all_emotion_idx, dtype=torch.long)
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks, 
                all_token_type_ids,
                all_utterance_input_ids, 
                all_utterance_token_type_ids, 
                all_utterance_mask, 
                all_emotion, 
                all_emotion_idx, 
                all_feature_index, 
                all_cls_index, 
                all_p_mask,
            )

        if not is_training:
            return dataset, examples, features
        return dataset 
        

    def train_fn(self, df_train, df_valid):
        
        train_dataset = self.load_and_cache_examples(df_train)
        train_data_sampler = RandomSampler(train_dataset)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_data_sampler, 
            batch_size = self.model_args.train_batch_size,
            num_workers=0
        )

        valid_dataset, examples, features = load_and_cache_examples(
            df_valid, evaluate=True, output_examples=True, model_args=self.model_args
        )

        valid_data_sampler = SequentialSampler(valid_dataset)
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            sampler=valid_data_sampler,
            batch_size = self.model_args.valid_batch_size,
            num_workers=0
        )
        
        valid_predictions = self.emotion_predictor.train(self.device, train_data_loader, valid_data_loader, len(df_train))
        
        # after EP training, we predict emotion and then again create a new dataloader
        # TODO: create a dataloader with emotions at emotion_idx
        '''
        print(len(valid_dataset))
        print("#"*20)
        # TODO: need to update question text with exact emotion
        print(len(examples))
        print("#"*20)
        # TODO: need to update input ids with exact emotion
        print(len(features))
        '''

        train_dataset = self.modify_dataset(train_dataset, is_training=True)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, 
            sampler=train_data_sampler,
            batch_size = self.model_args.train_batch_size,
            num_workers=0
        )

        # print(len(valid_dataset), len(examples))
        # breakpoint()
        valid_dataset, examples, features = self.modify_dataset(valid_dataset, predictions=valid_predictions, examples=examples, features=features, is_training=False)
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            sampler=valid_data_sampler,
            batch_size=self.model_args.valid_batch_size,
            num_workers=0
        )

        self.cause_span_predictor.train(self.device, train_data_loader, valid_data_loader, examples, features, len(df_train))

    def eval_fn(self, df_test):

        test_dataset, examples, features = load_and_cache_examples(
            df_test, evaluate=True, output_examples=True, model_args=self.model_args
        )
        test_data_sampler = SequentialSampler(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, 
            sampler=test_data_sampler,
            batch_size=self.model_args.valid_batch_size,
            num_workers=0
        )
        
        predictions = self.emotion_predictor.evaluate(self.device, test_data_loader)
        test_dataset, examples, features = self.modify_dataset(test_dataset, predictions, examples, features, is_training=False)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size = self.model_args.valid_batch_size,
            num_workers=0
        )
        results, text = self.cause_span_predictor.evaluate(test_data_loader, examples, features, config.TEST_DATASET, self.device)
        
        return results, text
