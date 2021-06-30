import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
from transformers import (
    BertModel,
)
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from question_answering_utils import *
import numpy as np
import config
from sklearn import metrics

from gensim.models import Word2Vec
import torch.nn as nn

from transformers import (
    BertForPreTraining, 
    BertPreTrainedModel,
)
from bert_utils import *
import pandas as pd

class BertCSModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    def add_emotion_vector(
        self, 
        embedding_output, 
        emotion_vector,
        emotion_idx, 
    ):
        embedding_output[:, emotion_idx, :] = emotion_vector
        return embedding_output


    def forward(
        self,
        emotion_vector=None,
        emotion_idx=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        # print(embedding_output.device)

        # embedding output shape: (batch_size, seq_len, 768)
        embedding_output = self.add_emotion_vector(embedding_output, emotion_vector,emotion_idx)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            # past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class E2E_Model(nn.Module):
    def __init__(self, device):
        super(E2E_Model, self).__init__()
        self.e_bert = BertModel.from_pretrained(config.emotion_model_name, cache_dir=config.TRANSFORMER_CACHE, config=config.EMOTION_CONFIG)
        self.e_linear = nn.Linear(768, config.n_classes)
        
        self.emotions = list(config.emotion_mapping.keys())

        ''' Creating and storing pretrained Glove weights
        # glove_dict = self.get_glove_dict()
        # self.weights, words = self.create_weights_matrix(glove_dict)

        # ['happiness', 'surprise', 'anger', 'sadness', 'disgust', 'fear', 'excited']
        # np.save('glove_emotion_weights.npy', self.weights)
        '''
        
        self.weights = np.load('glove_emotion_weights.npy')
        self.weights = torch.FloatTensor(self.weights).to(device)
        # model = Word2Vec(self.emotions, min_count=1, size=768) # vector size inplace of size for newer version
        # self.weights = torch.tensor(model.wv.vectors, requires_grad=True, device=device)  # 7 x 768
        
        self.embedding = nn.Embedding.from_pretrained(self.weights, freeze=True) # Initialize embeddings from Glove (size = 7 x 300)
        
        self.linear = nn.Linear(300, 768)
        # self.c_bert = # include code for all layers of bert, need to check the implementation on huggingface
        self.c_bert = BertCSModel.from_pretrained(config.cause_model_name, cache_dir=config.TRANSFORMER_CACHE, config=config.CAUSE_CONFIG)

        bert_out_size, emb_out_size, out_size = 768, self.weights.size(1), 2
        self.c_linear = nn.Linear(bert_out_size, out_size)

        torch.nn.init.normal_(self.c_linear.weight, std=0.02)
        torch.nn.init.normal_(self.e_linear.weight, std=0.02)
    
    
    def get_glove_dict(self):
        glove_dict = {}
        with open("./word2vec/glove.6B.300d.txt", "r", encoding="utf-8") as f:
            for line in f:
                values = line.split(' ')
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                glove_dict[word] = vector

        f.close()
        
        return glove_dict
    
    def create_weights_matrix(self, glove_dict):
        weights_matrix = np.zeros((len(self.emotions), 300))
        words_found = 0
        for i, word in enumerate(self.emotions):
            try:
                weights_matrix[i] = glove_dict[word]
                words_found += 1
            except:
                pass

        return weights_matrix, words_found

    def get_emotion_feat(self, utterance_input_ids, utterance_mask, utterance_token_type_ids, emotion):
        e_outputs = self.e_bert(
            input_ids=utterance_input_ids, 
            attention_mask=utterance_mask, 
            token_type_ids=utterance_token_type_ids)    # batch_size x 768
        
        linear_out = self.e_linear(e_outputs[1]) # 1 is the pooled output , batch_size x 7
        
        soft_out = F.softmax(linear_out, dim=1)
        weights_out = self.linear(self.weights)
        emotion_vector = torch.matmul(soft_out, weights_out) # batch_size x 768
        
        if emotion is not None:
            loss_fn = CrossEntropyLoss()
            emotion_loss = loss_fn(linear_out, emotion)
            return e_outputs[1], emotion_vector, torch.argmax(soft_out, dim=1), emotion_loss 
        
        return e_outputs[1], emotion_vector, torch.argmax(soft_out, dim=1)
        

    def forward(self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
        utterance_input_ids=None,
        utterance_mask=None,
        utterance_token_type_ids=None,
        emotion_idx=None,
        emotion=None,
    ):
        if emotion is not None:
            e_out, emotion_vector, em, emotion_loss = self.get_emotion_feat(utterance_input_ids, utterance_mask, utterance_token_type_ids, emotion)
        else:
            e_out, emotion_vector, em = self.get_emotion_feat(utterance_input_ids, utterance_mask, utterance_token_type_ids, emotion)

        embedding_bert = self.c_bert(
            emotion_vector=e_out,
            emotion_idx=emotion_idx,   # get the emotion idx
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_out = embedding_bert[0] # batch_size, 512, 768
        # emotion_vector = emotion_vector.unsqueeze(1)
        # out = torch.cat((seq_out, emotion_vector), dim=1)
        
        c_linear_out = self.c_linear(seq_out)

        start_logits, end_logits = c_linear_out.split(1, dim=-1)  # (batch_size, 513, 1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None

        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)  # clamp between 0 and 513
            end_positions.clamp_(0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss)/2 + emotion_loss 

            
        return CustomQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=embedding_bert.hidden_states,
            attentions=embedding_bert.attentions,
            p_emotion=em,
        )

class RecModel(nn.Module):
    def __init__(self, is_emotion_training=False, device='cuda', best_model=None):
        super(RecModel, self).__init__()
        self.model = E2E_Model(device)
        self.device = device
        self.path = best_model

    def _get_inputs_dict(self, batch, is_training=True):
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

        return inputs
        
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
            max_seq_length=config.MAX_CLENGTH,
            doc_stride=256,
            max_query_length=config.MAX_QLENGTH,
            is_training=not evaluate,
            # tqdm_enabled=not args.silent,
            # threads=args.process_count,
            # args=args,
        )
        
        if output_examples:
            return dataset, examples, features
        return dataset
    
    def train(self, data_loader, model, optimizer, device, scheduler):
        model.train()

        tk0 = tqdm(data_loader, total=len(data_loader), desc='E2E Training')
        for bi, d in enumerate(tk0):
            inputs = self._get_inputs_dict(d, is_training=True)

            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            mask = inputs['attention_mask']
            targets_start = inputs['start_positions']
            targets_end = inputs['end_positions']
            emotion = inputs['emotion']
            emotion_idx = inputs['emotion_idx']
            utterance_input_ids = inputs['utterance_input_ids']
            utterance_mask = inputs['utterance_mask']
            utterance_token_type_ids = inputs['utterance_token_type_ids']
            emotion = inputs['emotion']

            input_ids = input_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)
            utterance_input_ids = utterance_input_ids.to(device, dtype=torch.long)
            utterance_mask = utterance_mask.to(device, dtype=torch.long)
            utterance_token_type_ids = utterance_token_type_ids.to(device, dtype=torch.long)
            emotion = emotion.to(device, dtype=torch.long)

            outputs = model(
                input_ids=input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=mask, 
                start_positions=targets_start,
                end_positions=targets_end,
                utterance_input_ids=utterance_input_ids,
                utterance_mask=utterance_mask,
                utterance_token_type_ids=utterance_token_type_ids,
                emotion_idx=emotion_idx,
                emotion=emotion,
            )

            loss = outputs[0]
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            scheduler.step()
            
            
            '''
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            '''

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

        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
        )
    
        # model = nn.DataParallel(model) # for multiple GPUs
        best_eval_loss = 1e8
        monitor_eval_loss = []
        for epoch in range(config.EPOCHS):
            # print("..............Inside cause Epoch...............")
            self.train(train_data_loader, model, optimizer, self.device, scheduler)
            
            # TODO: Look at what does squad use for evaluation
            all_predictions, all_nbest_json, scores_diff_json, eval_loss, actual_emotion, predicted_emotion = self.eval(valid_data_loader, examples, features, model, self.device)

            df_valid = pd.read_csv(config.VALID_DATASET)
            truth = df_valid[['id', 'cause_span']]

            result, texts = self.calculate_results(truth, all_predictions)
            result["eval_loss"] = eval_loss

            monitor_eval_loss.append(eval_loss)
            # return result, texts
            
            if eval_loss < best_eval_loss:
                self.save_model(self.path, optimizer, scheduler, model=model, eval_loss=eval_loss)
                best_eval_loss = eval_loss
            
            print("Epoch: {}\t Lr: {:.6f}\t Val Loss: {:.4f}\t".format(epoch, get_lr(optimizer), eval_loss))

    def eval(self, data_loader, examples, features, model, device):
        all_results = []
        actual_emotion = []
        predicted_emotion = []

        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        
        for bi, d in enumerate(tqdm(data_loader, total=len(data_loader), desc='E2E Evaluation')):
            with torch.no_grad():
                inputs = self._get_inputs_dict(d, is_training=False)

                input_ids = inputs['input_ids']
                token_type_ids = inputs['token_type_ids']
                mask = inputs['attention_mask']
                emotion = inputs['emotion']
                emotion_idx = inputs['emotion_idx']
                utterance_input_ids = inputs['utterance_input_ids']
                utterance_mask = inputs['utterance_mask']
                utterance_token_type_ids = inputs['utterance_token_type_ids']
                # print(inputs)

                # input_ids, token_type_ids, mask = self.modify_dataset_with_emotions(input_ids, token_type_ids, mask, emotion_idx, emotion)

                input_ids = input_ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                utterance_input_ids = utterance_input_ids.to(device, dtype=torch.long)
                utterance_mask = utterance_mask.to(device, dtype=torch.long)
                utterance_token_type_ids = utterance_token_type_ids.to(device, dtype=torch.long)

                # print(example_indices)
                outputs = model(
                    input_ids=input_ids, 
                    token_type_ids=token_type_ids, 
                    attention_mask=mask, 
                    utterance_input_ids=utterance_input_ids,
                    utterance_mask=utterance_mask,
                    utterance_token_type_ids=utterance_token_type_ids,
                    emotion_idx=emotion_idx,
                    
                )

                actual_emotion.extend(emotion.cpu().detach().numpy().tolist())
                predicted_emotion.extend(outputs['p_emotion'].cpu().detach().numpy().tolist())

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
        # print(all_results)

        all_predictions, all_nbest_json, scores_diff_json = self.write_predictions(
            examples,
            features,
            all_results,
            20,
            config.MAX_ALENGTH,
            False,
            True,
            0.0,
        )

        return all_predictions, all_nbest_json, scores_diff_json, eval_loss, actual_emotion, predicted_emotion

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

        ckpt = torch.load(os.path.join(self.path, 'best_epoch.pth'))
        self.model.load_state_dict(ckpt['model_state_dict'])
        model = self.model.to(self.device)
        
        all_predictions, all_nbest_json, scores_diff_json, eval_loss, actual_emotion, predicted_emotion = self.eval(test_data_loader, examples, features, model, self.device)
        
        df_valid = pd.read_csv(config.TEST_DATASET)
        truth = df_valid[['id', 'cause_span']]

        result, texts = self.calculate_results(truth, all_predictions)
        result["eval_loss"] = eval_loss

        return result, texts, actual_emotion, predicted_emotion
        

    def calculate_results(self, truth, predictions, **kwargs):
        truth_dict = truth.set_index('id').T.to_dict('list').copy()
        truth_dict = {k: v[0] for k, v in truth_dict.items()}
        # print(truth_dict)
        # print()
        # print(predictions)

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
                start_indexes = get_best_indexes(result.start_logits, n_best_size)
                end_indexes = get_best_indexes(result.end_logits, n_best_size)
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

            probs = compute_softmax(total_scores)

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

        return all_predictions, all_nbest_json, scores_diff_json

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, eval_loss=None):
        save_path = os.path.join(output_dir, "best_epoch.pth")
        state = {
            'eval_loss': eval_loss,
			'model_state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
	     
        torch.save(state, save_path)