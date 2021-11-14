import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import os

from tqdm import tqdm
from transformers import (
    BertModel,
    AutoModel,
    
)
import torch
import collections
from transformers import AdamW, get_linear_schedule_with_warmup

import wandb
import dataset 
from utils import *
import pandas as pd
import numpy as np
import config
from sklearn import metrics

from question_answering_utils import (
    load_and_cache_examples, 
    _get_best_indexes,
    get_final_text,
    _compute_softmax,

)

RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

class CauseSpanModel(nn.Module):
    def __init__(self):
        super(CauseSpanModel, self).__init__()
        self.bert = AutoModel.from_pretrained(config.cause_model_name, config=config.CAUSE_CONFIG, cache_dir=config.TRANSFORMER_CACHE)
        # self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config.CAUSE_CONFIG.hidden_size, 2)
    
    def forward(self, 
        input_ids=None, 
        token_type_ids=None, 
        attention_mask=None, 
        start_positions=None, 
        end_positions=None
    ):
        outputs = self.bert(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        
        seq_out = outputs[0]
        logits = self.linear(seq_out)
        
    
        start_logits, end_logits = logits.split(1, dim=-1)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
class CauseSpanPredictor:
    def __init__(self, path, model_args):
        self.model = CauseSpanModel()
        self.path = path
        self.model_args = model_args

    def _get_inputs_dict(self, batch):
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
        return inputs

    def loss_function(self, start_logits, end_logits, start_positions, end_positions):
        loss_func = nn.CrossEntropyLoss()
        start_loss = loss_func(start_logits, start_positions)
        end_loss = loss_func(end_logits, end_positions)
        total_loss = (start_loss + end_loss)
        return total_loss
    
    def train_fn(self, data_loader, model, optimizer, device, scheduler):
        model.train()
        train_loss = AverageMeter()

        for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc='Cause Span Training')):
            inputs = self._get_inputs_dict(d)

            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            mask = inputs['attention_mask']
            targets_start = inputs['start_positions']
            targets_end = inputs['end_positions']
            emotion = inputs['emotion']
            emotion_idx = inputs['emotion_idx']

            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            model.zero_grad()

            outputs = model(
                input_ids=input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=mask, 
                start_positions=targets_start,
                end_positions=targets_end,
            )
            
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.update(loss.item(), targets_start.size(0))
            
        return train_loss.avg

    def train(self, device, train_data_loader, valid_data_loader, examples, features, train_len):
        model = self.model
        model = model.to(device)
        
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

        num_train_steps = int(train_len/self.model_args.train_batch_size*self.model_args.epochs)

        optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=num_train_steps
        )
    
        train_loss_list = []
        eval_loss_list = []
        best_eval_loss = 1e8
        best_score = -1
        best_f1 = 0

        for epoch in range(self.model_args.epochs):
            train_loss_num = self.train_fn(train_data_loader, model, optimizer, device, scheduler)
            
            all_predictions, all_nbest_json, scores_diff_json, eval_loss, eval_loss_num = self.eval_fn(valid_data_loader, examples, features, model, device)
            
            train_loss_list.append(train_loss_num)
            eval_loss_list.append(eval_loss_num)

            df_valid = pd.read_csv(config.VALID_DATASET)
            truth = df_valid[['id', 'cause_span']]

            result, texts = self.calculate_results(truth, all_predictions)
            result["eval_loss"] = eval_loss

            r, (exact, pos_f1) = evaluate_results(texts)
            if config.WANDB: 
                wandb.log({"cs_train_loss": train_loss_num, "cs_val_loss": eval_loss_num, \
                "cs_cum_val_loss": eval_loss, "exact_match":exact, "Pos_F1":pos_f1})

            if pos_f1 > best_f1:
                save_path = 'cause_epoch_{}.pth'.format(epoch)
                torch.save(self.model.state_dict(), os.path.join(self.path, save_path))
                best_f1 = pos_f1 

        
    def eval_fn(self, data_loader, examples, features, model, device):
        all_results = []
        eval_loss = 0.0
        nb_eval_steps = 0
        eval_loss_list = AverageMeter()

        model.eval()

        for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc='Cause Span Evaluation')):
            with torch.no_grad():
                inputs = self._get_inputs_dict(d)

                input_ids = inputs['input_ids']
                token_type_ids = inputs['token_type_ids']
                mask = inputs['attention_mask']
                targets_start = inputs['start_positions']
                targets_end = inputs['end_positions']
                emotion = inputs['emotion']
                emotion_idx = inputs['emotion_idx']


                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)

                outputs = model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids, 
                    attention_mask=mask, 
                )
                
                eval_loss_list.update(outputs[0].mean().item(), targets_start.size(0))
                eval_loss += outputs[0].mean().item()
                example_indices = d[8]  # all feature index

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
            
                    result = RawResult(
                        unique_id=unique_id,
                        start_logits=to_list(outputs[0][i]),
                        end_logits=to_list(outputs[1][i]),
                    )

                    all_results.append(result)

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        all_predictions, all_nbest_json, scores_diff_json = self.write_predictions(
            examples,
            features,
            all_results,
            20,
            self.model_args.max_alength,
            False,
            True,
            0.0,
        )

        return all_predictions, all_nbest_json, scores_diff_json, eval_loss, eval_loss_list.avg

    def evaluate(self, data_loader, examples, features, eval_data, device):
        model = self.model
        model.load_state_dict(torch.load(self.path))
        model = model.to(device)
        all_predictions, all_nbest_json, scores_diff_json, eval_loss, eval_loss_list = self.eval_fn(data_loader, examples, features, model, device)

        df_valid = pd.read_csv(eval_data)
        truth = df_valid[['id', 'cause_span']]

        result, texts = self.calculate_results(truth, all_predictions)
        result["eval_loss"] = eval_loss

        return result, texts

    def calculate_results(self, truth, predictions, **kwargs):
        truth_dict = truth.set_index('id').T.to_dict('list').copy()
        truth_dict = {k: v[0] for k, v in truth_dict.items()}

        correct = 0
        incorrect = 0
        similar = 0
        correct_text = {}
        incorrect_text = {}
        similar_text = {}
        predicted_answers = []
        true_answers = []

        for q_id, pred_answer in predictions.items():
            answer = truth_dict[q_id]
            predicted_answers.append(pred_answer)
            true_answers.append(answer)
            if predictions[q_id].strip() == answer.strip():
                correct += 1
                correct_text[q_id] = answer
            elif predictions[q_id].strip() in answer.strip() or answer.strip() in predictions[q_id].strip():
                similar += 1
                similar_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                }
            else:
                incorrect += 1
                incorrect_text[q_id] = {
                    "truth": answer,
                    "predicted": predictions[q_id],
                }

        extra_metrics = {}
        for metric, func in kwargs.items():
            extra_metrics[metric] = func(true_answers, predicted_answers)

        result = {"correct": correct, "similar": similar, "incorrect": incorrect, **extra_metrics}

        texts = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }
        return result, texts

    def write_predictions(
        self,
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative,
        null_score_diff_threshold,
    ):
        """Write final predictions to the json file and log-odds of null if needed."""
        # logger.info("Writing predictions to: %s" % (output_prediction_file))
        # logger.info("Writing nbest to: %s" % (output_nbest_file))

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)
        
        # print(example_index_to_features)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"],
        )

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()
        scores = []

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # the start logit at the slice with min null score
            null_end_logit = 0  # the end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                scores.append({'tokens': feature.tokens, 'start': result.start_logits, 'end': result.end_logits})
                start_indexes = _get_best_indexes(result.start_logits, n_best_size)
                end_indexes = _get_best_indexes(result.end_logits, n_best_size)
                # if we could have irrelevant answers, get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions, e.g., predict
                        # that the start of the span is in the question. We throw out all
                        # invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                            )
                        )
            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True,)
            # print(prelim_predictions)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "start_logit", "end_logit"]
            )

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                if pred.start_index > 0:  # this is a non-null prediction
                    feature = features[pred.feature_index]
                    tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit,))
            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

                # In very rare edge cases we could only have single null prediction.
                # So we just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json
    
        # with open(output_prediction_file, "w") as writer:
        #     writer.write(json.dumps(all_predictions, indent=4) + "\n")

        # with open(output_nbest_file, "w") as writer:
        #     writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        # if version_2_with_negative:
        #     with open(output_null_log_odds_file, "w") as writer:
        #         writer.write(json.dumps(scores_diff_json, indent=4) + "\n")
        
        with open('scores.npy', 'wb') as file:
            np.save(file, scores)
        # exit()
        return all_predictions, all_nbest_json, scores_diff_json

