from transformers import (
    BertTokenizer, 
    BertConfig,
    AutoTokenizer,
    AutoModel,
)
import math

WANDB = False
SEED = 42

BEAM_SIZE=6
EPOCHS = 5
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
MAX_QLENGTH = 512
MAX_ALENGTH = 200
MAX_CLENGTH = 512
MAX_ULENGTH = 200
LEARNING_RATE = 4e-5 
HIGH_DROPOUT = 0.5
N_LAST_HIDDEN = 12
LOWER_CASE = False   # do_lower_case in get_final_text
BETA = 1  # emotion loss
DOC_STRIDE = 512

TRAIN_DATASET = 'data/fold1/dailydialog_qa_train_without_context.csv'
VALID_DATASET = 'data/fold1/dailydialog_qa_valid_without_context.csv'
TEST_DATASET = 'data/fold1/dailydialog_qa_test_without_context.csv'
# TEST_DATASET = 'data/fold1/iemocap_qa_test_without_context.csv'

TRANSFORMER_CACHE = '~/.cache'

emotion_model_name = 'bert-base-uncased'
EMOTION_CONFIG = BertConfig.from_pretrained(emotion_model_name, cache_dir=TRANSFORMER_CACHE)
EMOTION_TOKENIZER = BertTokenizer.from_pretrained(emotion_model_name, config=EMOTION_CONFIG, cache_dir=TRANSFORMER_CACHE)

cause_model_name = 'mrm8488/spanbert-finetuned-squadv2'
CAUSE_CONFIG = BertConfig.from_pretrained(cause_model_name, cache_dir=TRANSFORMER_CACHE, output_hidden_states=True)
CAUSE_TOKENIZER = BertTokenizer.from_pretrained(cause_model_name, config=CAUSE_CONFIG, cache_dir=TRANSFORMER_CACHE)

parameter_dict = { 
    'learning_rate': {
        'values': [1e-5, 4e-5, 7e-5, 5e-4]
    },
    
}

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'    
    },
    'parameters': parameter_dict,

}

emotion_mapping = {
    'happiness': 0, 
    'surprise': 1,
    'anger': 2,
    'sadness': 3,
    'disgust': 4,
    'fear': 5,
}

n_classes = len(emotion_mapping)