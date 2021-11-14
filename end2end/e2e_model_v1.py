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
from sklearn.metrics import classification_report
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
from beam import BeamStart, BeamEnd

class E2E_Model(nn.Module):
    def __init__(self, device, n_outputs=2):
        super(E2E_Model, self).__init__()
        config_class = config.CAUSE_CONFIG
        # self.e_bert = BertModel.from_pretrained(config.emotion_model_name, cache_dir=config.TRANSFORMER_CACHE, config=config.EMOTION_CONFIG)
        self.e_linear = nn.Linear(config_class.hidden_size, config.n_classes)
        
        # self.c_bert = # include code for all layers of bert, need to check the implementation on huggingface
        self.c_bert = BertModel.from_pretrained(config.cause_model_name, cache_dir=config.TRANSFORMER_CACHE, config=config.CAUSE_CONFIG)
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)

        n_hidden = config_class.hidden_size//2
        self.c_bilstm = nn.LSTM(config_class.hidden_size, n_hidden, batch_first=True, bidirectional=True)

        self.c_linear = nn.Linear(config_class.hidden_size*2, n_outputs)
        # self.neg_linear = nn.Linear(512, 2)
        # self.start_linear = nn.Linear(config.CAUSE_CONFIG.hidden_size, 1)
        # self.end_linear = nn.Linear(config.CAUSE_CONFIG.hidden_size+1, 1)
        self.start = BeamStart(config_class)
        self.end = BeamEnd(config_class)

        self.device = device

        torch.nn.init.normal_(self.c_linear.weight, std=0.02)
        torch.nn.init.normal_(self.e_linear.weight, std=0.02)

    def get_emotion_feat(self, utterance_input_ids, utterance_mask, utterance_token_type_ids):
        e_outputs = self.e_bert(
            input_ids=utterance_input_ids, 
            attention_mask=utterance_mask, 
            token_type_ids=utterance_token_type_ids)  
        
        linear_out = self.e_linear(e_outputs[1]) 
        return linear_out, e_outputs[1]

    def forward(self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None,
        utterance_input_ids=None,
        utterance_mask=None,
        utterance_token_type_ids=None,
        start_positions=None,
        end_positions=None,
        beam_size=config.BEAM_SIZE, 
        p_mask=None,
    ):
        
        # e_linear_out, e_pool = self.get_emotion_feat(utterance_input_ids, utterance_mask, utterance_token_type_ids)
        
        output = self.c_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        seq_out, pool_out, out = output[0], output[1], output[2] 
        # emotion_vector = emotion_vector.unsqueeze(1)
        # out = torch.cat((seq_out, emotion_vector), dim=1)

        o = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out = torch.mean(o, dim=0)
        # out_max, _ = torch.max(o, dim=0)
        # out = torch.cat((out, out_max), dim=-1)
        start_logits = self.start(out)
        # torch.hstack([start_logits, (start_positions==0).unsqueeze(1)])


        # e_pool = e_pool.unsqueeze(dim=1)
        # out = torch.cat([e_pool, out], dim=1)
        # out, hidden_states = self.c_bilstm(out)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        # c_linear_out = torch.mean(torch.stack([
        #         self.c_linear(self.high_dropout(out))
        #         for _ in range(5)
        #     ], dim=0), dim=0)
        
        e_linear_out = self.e_linear(out[:, 0, :])
        # c_linear_out = self.c_linear(out)
        if start_positions is not None and end_positions is not None: #training
            end_logits = self.end(out, start_positions=start_positions, p_mask=p_mask)
            return start_logits, end_logits, e_linear_out

        else: #inference
            bsz, slen, hsz = out.size()
            start_log_probs = F.softmax(start_logits, dim=-1)
            start_index_zero = start_log_probs[:, 0]
            
            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, beam_size, dim=-1
            )

            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)
            start_states = torch.gather(out, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)
            hidden_states_expanded = out.unsqueeze(2).expand_as(
                start_states
            )
            
            end_logits = self.end(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)
            end_index_zero = end_log_probs[:, 0, :]

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, beam_size, dim=1
            )
            
            end_top_log_probs = end_top_log_probs.view(-1, beam_size * beam_size)
            end_top_index = end_top_index.view(-1, beam_size * beam_size)
            
            
            # outputs = start_top_log_probs, start_top_index, end_top_log_probs, end_top_index 
            return start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, e_linear_out, start_index_zero, end_index_zero
        

class RecModel(nn.Module):
    def __init__(self, device='cuda', best_model=None, em_weights=None, neg_weights=None):
        super(RecModel, self).__init__()
        self.model = E2E_Model(device)
        self.em_weights = em_weights
        self.device = device
        self.example_cnt = 0
        self.batch_cnt = 0
        self.eval_batch_cnt = 0
        self.eval_example_cnt = 0

        self.path = best_model
        

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
        
    def load_and_cache_examples(self, examples, evaluate=False, no_cache=False, output_examples=False):
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
            max_seq_length=config.MAX_CLENGTH,
            doc_stride=config.DOC_STRIDE,
            max_query_length=config.MAX_QLENGTH,
            is_training=not evaluate,
            # tqdm_enabled=not args.silent,
            # threads=args.process_count,
            # args=args,
        )
        
        if output_examples:
            return dataset, examples, features
        return dataset

    def e_loss_fn(self, outputs, targets):
        weights = torch.tensor(self.em_weights, dtype=torch.float)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        class_weights = torch.FloatTensor(weights).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        emotion_loss = loss_fn(outputs, targets)
        return emotion_loss

      
    def c_loss_fn(self, start_logits, start_positions, end_logits, end_positions):
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)  # clamp between 0 and 512
        end_positions = end_positions.clamp(0, ignored_index)
        
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss)/2
        return total_loss
    
    def neg_loss_fn(self, neg_logit, neg_val):
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(neg_logit, neg_val.long())
        return loss


    def train_fn(self, df_train, df_valid):
        train_dataset = self.load_and_cache_examples(df_train)
        train_data_sampler = RandomSampler(train_dataset)

        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_data_sampler, 
            batch_size = config.TRAIN_BATCH_SIZE,
            num_workers=0
        )
        
        valid_dataset, examples, features = load_and_cache_examples(
            df_valid, evaluate=True, output_examples=True
        )

        valid_data_sampler = SequentialSampler(valid_dataset)
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            sampler=valid_data_sampler,
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

        num_train_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE*config.EPOCHS)

        optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=4,
            num_training_steps=num_train_steps
        )
        if config.WANDB:
            wandb.watch(model, log='all', log_freq=200)

        best_eval_loss = 1e8
        best_exact = -1
        best_f1 = -1
        for epoch in range(config.EPOCHS):
            train_loss_avg = self.train(train_data_loader, model, optimizer, self.device, scheduler)
            
            all_predictions, eval_loss_avg, e_acc, e_f1 = self.eval(valid_data_loader, examples, features, model, self.device)

            df_valid = pd.read_csv(config.VALID_DATASET)
            truth = df_valid[['id', 'cause_span']]

            result, texts = self.calculate_results(truth, all_predictions)
            r, (exact, pos_f1) = evaluate_results(texts)
            # result["eval_loss"] = eval_loss
            if config.WANDB:
                wandb.log({'train_loss': train_loss_avg, 'eval_loss': eval_loss_avg, \
                    'e_accuracy': e_acc, 'e_f1': e_f1, 'exact_match': exact, \
                        'pos_f1': pos_f1, 'epoch': epoch})
            print("Epoch Number: "+ str(epoch+1) + " done")
            if pos_f1 > best_f1:
                save_path = 'epoch{}.pth'.format(epoch)
                torch.save(self.model.state_dict(), os.path.join(self.path, save_path))
                best_exact = exact


    def train(self, data_loader, model, optimizer, device, scheduler):
        model.train()
        losses = AverageMeter()

        tk0 = tqdm(data_loader, total=len(data_loader), desc='Training')
        for bi, d in enumerate(tk0):
            inputs = self._get_inputs_dict(d)

            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            mask = inputs['attention_mask']
            start_positions = inputs['start_positions']
            end_positions = inputs['end_positions']
            emotion = inputs['emotion']
            emotion_idx = inputs['emotion_idx']
            utterance_input_ids = inputs['utterance_input_ids']
            utterance_mask = inputs['utterance_mask']
            utterance_token_type_ids = inputs['utterance_token_type_ids']

            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            start_positions = start_positions.to(device, dtype=torch.long)
            end_positions = end_positions.to(device, dtype=torch.long)
            utterance_input_ids = utterance_input_ids.to(device, dtype=torch.long)
            utterance_mask = utterance_mask.to(device, dtype=torch.long)
            utterance_token_type_ids = utterance_token_type_ids.to(device, dtype=torch.long)
            emotion = emotion.to(device, dtype=torch.long)
            
            model.zero_grad()
            # with torch.autograd.set_detect_anomaly(True):
            start_logits, end_logits, e_linear_out = model(
                input_ids=input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=mask, 
                utterance_input_ids=utterance_input_ids,
                utterance_mask=utterance_mask,
                utterance_token_type_ids=utterance_token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            c_loss = self.c_loss_fn(start_logits, start_positions, end_logits, end_positions)
            e_loss = self.e_loss_fn(e_linear_out, emotion)

            loss = c_loss + config.BETA * e_loss 
            loss = c_loss
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            self.example_cnt += input_ids.size(0)
            self.batch_cnt += 1
            if config.WANDB:
                if ((self.batch_cnt + 1) % 100) == 0:
                    wandb.log({'train_loss_step': loss, 'example_count:': self.example_cnt})
                    #   plot_grad_flow(model.named_parameters(), './viz_gradients', self.example_cnt)

            losses.update(loss.item(), input_ids.size(0))
            tk0.set_postfix(loss=losses.avg)
        
        return losses.avg

    def eval(self, data_loader, examples, features, model, device):
        model.eval() 
        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_top_log_probs", "start_top_index", "end_top_log_probs", "end_top_index", "start_index_zero", "end_index_zero"])  
        all_results = []
        eval_loss = 0.0
        nb_eval_steps = 0
        e_targets = []
        e_outputs = []
        losses = AverageMeter()
        
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), desc='Evaluation')
            for bi, d in enumerate(tk0):
                inputs = self._get_inputs_dict(d)

                input_ids = inputs['input_ids']
                token_type_ids = inputs['token_type_ids']
                mask = inputs['attention_mask']
                start_positions = inputs['start_positions']
                end_positions = inputs['end_positions']
                emotion = inputs['emotion']
                emotion_idx = inputs['emotion_idx']
                utterance_input_ids = inputs['utterance_input_ids']
                utterance_mask = inputs['utterance_mask']
                utterance_token_type_ids = inputs['utterance_token_type_ids']

                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                start_positions = start_positions.to(device, dtype=torch.long)
                end_positions = end_positions.to(device, dtype=torch.long)
                utterance_input_ids = utterance_input_ids.to(device, dtype=torch.long)
                utterance_mask = utterance_mask.to(device, dtype=torch.long)
                utterance_token_type_ids = utterance_token_type_ids.to(device, dtype=torch.long)
                em_targets = emotion.to(device, dtype=torch.long)
                
                # print(example_indices)
                start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, e_linear_out, start_index_zero, end_index_zero = model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids, 
                    attention_mask=mask, 
                    utterance_input_ids=utterance_input_ids,
                    utterance_mask=utterance_mask,
                    utterance_token_type_ids=utterance_token_type_ids,
                )
                
                start_top_log_probs = start_top_log_probs.detach().cpu().numpy()
                start_top_index = start_top_index.detach().cpu().numpy()
                end_top_log_probs = end_top_log_probs.detach().cpu().numpy()
                end_top_index = end_top_index.detach().cpu().numpy()
                start_index_zero = start_index_zero.detach().cpu().numpy()
                end_index_zero = end_index_zero.detach().cpu().numpy()

                # end_logits_prob = end_logits
                # end_logits = end_logits.split(1, dim=-1)[0].squeeze(-1)
                # # print(end_logits.size())
                # c_loss = self.c_loss_fn(start_logits, start_positions, end_logits, end_positions)
                # e_loss = self.e_loss_fn(e_linear_out, emotion)
                # loss = c_loss + e_loss
                
                example_indices = d[10]  # all feature index

                for i, example_index in enumerate(example_indices):
                    eval_feature = features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    result = RawResult(
                        unique_id=unique_id,
                        start_top_log_probs=start_top_log_probs[i],
                        start_top_index=start_top_index[i],
                        end_top_log_probs=end_top_log_probs[i].reshape(config.BEAM_SIZE, config.BEAM_SIZE),
                        end_top_index=end_top_index[i].reshape(config.BEAM_SIZE, config.BEAM_SIZE),
                        start_index_zero = start_index_zero[i],
                        end_index_zero = end_index_zero[i],
                    )
                    all_results.append(result)

                if 'iemocap' in config.TEST_DATASET:
                    e_linear_out = e_linear_out[:, [0, 2, 3, 6]]
                    dd_to_iemo = {
                        0: 0,
                        2: 1,
                        3: 2,
                        6: 3 
                    }
                    em_targets_cpu = em_targets.cpu().detach().numpy().tolist()
                    em_targets_cpu = [dd_to_iemo[x] for x in em_targets_cpu]
                    e_targets.extend(em_targets_cpu)
                    e_outputs.extend(torch.argmax(e_linear_out, axis=1).cpu().detach().numpy().tolist())
                else:
                    e_targets.extend(em_targets.cpu().detach().numpy().tolist())
                    e_outputs.extend(torch.argmax(e_linear_out, axis=1).cpu().detach().numpy().tolist())

                self.eval_batch_cnt += 1
                self.eval_example_cnt += input_ids.size(0)

                # if config.WANDB:
                #     if ((self.eval_batch_cnt + 1) % 100) == 0:
                #         wandb.log({'eval_loss_step': loss, 'eval_example_count': self.eval_example_cnt})
                        #   plot_grad_flow(model.named_parameters(), './viz_gradients', self.example_cnt)

                # losses.update(loss.item(), input_ids.size(0))
                # tk0.set_postfix(loss=losses.avg)
        all_predictions = self.write_predictions(
            all_examples=examples,
            all_features=features,
            all_results=all_results,
            max_answer_length=config.MAX_ALENGTH,
        )

        e_accuracy = metrics.accuracy_score(e_targets, e_outputs)
        e_f1 = metrics.f1_score(e_targets, e_outputs, average='macro')
        print("Emotion Confusion matrix")
        print(metrics.confusion_matrix(e_targets, e_outputs))
        
        return all_predictions, losses.avg, e_accuracy, e_f1

    def eval_fn(self, df_test):
        test_dataset, examples, features = load_and_cache_examples(
            df_test, evaluate=True, output_examples=True
        )

        test_data_sampler = SequentialSampler(test_dataset)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, 
            sampler=test_data_sampler,
            batch_size = config.VALID_BATCH_SIZE,
            num_workers=0
        )

        self.model.load_state_dict(torch.load(self.path), strict=False)
        self.model = self.model.to(self.device)
        
        all_predictions, eval_loss_avg, acc, f1 = self.eval(test_data_loader, examples, features, self.model, self.device)
        # df_valid = pd.read_csv(config.TEST_DATASET)
        df_valid = df_test
        truth = df_valid[['id', 'cause_span']]

        result, texts = self.calculate_results(truth, all_predictions)
        print("Emotion Accuracy:", acc)
        print("Emotion F1:", f1)
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
            elif predictions[q_id].strip() in answer.strip() or answer.strip() in predictions[q_id].strip() or (predictions[q_id] == '' and answer=="Impossible \u2260\u00e6\u2260\u00e6\u2260 answer"):
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

        
        result = {"correct": correct, "similar": similar, "incorrect": incorrect}

        texts = {
            "correct_text": correct_text,
            "similar_text": similar_text,
            "incorrect_text": incorrect_text,
        }
        
        true_answer_new = ['' if x=="Impossible \u2260\u00e6\u2260\u00e6\u2260 answer" else x for x in true_answers]
        pred_neg = np.array([len(x) for x in predicted_answers]) > 0
        truth_neg = np.array([len(x) for x in true_answer_new]) > 0
        res = classification_report(truth_neg, pred_neg, target_names=['Negative', 'Positive'], digits=4)
        print("Classification of Negative Samples")
        print(res)

        return result, texts


    def write_predictions(
        self, 
        all_examples,
        all_features,
        all_results,
        max_answer_length,
        beam_size=config.BEAM_SIZE,
        null_score_diff_threshold=0.0
    ):
        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple( 
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_prob", "end_prob"],
        )

        all_predictions = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]
            prelim_predictions = []
            score_null = 100000  # large possible value
            min_null_feature_index = 0
            null_start_prob = 0
            null_end_prob = 0
            
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                ### null score
                for end_i in result.end_index_zero:
                    feature_null_score = result.start_index_zero + end_i
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_prob = result.start_index_zero
                        null_end_prob = end_i
                
                valid_start=len(feature.tokens) - feature.paragraph_len - 1
                valid_end=len(feature.tokens)
                # best = (valid_start, valid_end - 1)
                best = (0, 0)   # for negative samples
                best_score = -10000

                for i in range(len(result.start_top_log_probs)):
                    for j in range(result.end_top_log_probs.shape[0]):
                        if result.start_top_index[i] >= len(feature.tokens):
                            continue
                        if result.end_top_index[j,i]  >= len(feature.tokens):
                            continue
                        if result.start_top_index[i] not in feature.token_to_orig_map:  #  ??
                            continue
                        if result.end_top_index[j,i] not in feature.token_to_orig_map:    # ??
                            continue
                        if not feature.token_is_max_context.get(result.start_top_index[i], False):  # ??
                            continue
                        if result.end_top_index[j,i] < result.start_top_index[i]:
                            continue
                        length = result.end_top_index[j,i] - result.start_top_index[i] + 1
                        if length > max_answer_length:
                            continue
                        
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=result.start_top_index[i],
                                end_index=result.end_top_index[j,i],
                                start_prob=result.start_top_log_probs[i],
                                end_prob=result.end_top_log_probs[j, i],
                            )
                        )

                        # if valid_start <= result.start_top_index[i] < valid_end and valid_start <= result.end_top_index[j,i] < valid_end and result.start_top_index[i] <= result.end_top_index[j,i]:
                        #     score = result.start_top_log_probs[i] * result.end_top_log_probs[j,i]
                        #     if score > best_score:
                        #         best = (result.start_top_index[i], result.end_top_index[j,i])
                        #         best_score = score

                
                # start_index, end_index = best
                # if valid_start <= start_index < valid_end and valid_start <= end_index < valid_end and start_index < end_index:
                #     pass
                # else:
                #     start_index, end_index = 0, 0
                    
                

                # prelim_predictions.append(
                #     _PrelimPrediction(
                #         feature_index=feature_index,
                #         start_index=start_index,
                #         end_index=end_index,
                #     )
                # ) 
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_prob=null_start_prob,
                    end_prob=null_end_prob,
                )
            )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_prob + x.end_prob), reverse=True,)

            _NbestPrediction = collections.namedtuple(  
                "NbestPrediction", ["text", "start_prob", "end_prob"]
            )
            seen_predictions = {}
            nbest = []

            for pred in prelim_predictions:
                # if pred.neg_val == 0:
                #     final_text = ""
                #     seen_predictions[final_text] = True
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

                    final_text = get_final_text(tok_text, orig_text, do_lower_case=config.LOWER_CASE)
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text] = True
                
                else:
                    final_text = ""
                    seen_predictions[final_text] = True
                
                nbest.append(_NbestPrediction(text=final_text, start_prob=pred.start_prob, end_prob=pred.end_prob))  

            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_prob=null_start_prob, end_prob=null_end_prob))
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_prob=0.0, end_prob=0.0))
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_prob=0.0, end_prob=0.0))

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_prob + entry.end_prob)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                nbest_json.append(output)

            score_diff = score_null - best_non_null_entry.start_prob - (best_non_null_entry.end_prob)
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        # breakpoint()
        return all_predictions
            

    