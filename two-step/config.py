from transformers import (
    BertTokenizer, 
    BertConfig,
    BertTokenizerFast,
    AutoModel,
    AutoConfig,
    AutoTokenizer,

)

WANDB = False
SEED = 25
LEARNING_RATE = 4e-5
FOLD = 'fold1'
TRAIN_DATASET = '../sub1_data/fold1/dailydialog_qa_train_without_context.csv'
VALID_DATASET = '../sub1_data/fold1/dailydialog_qa_valid_without_context.csv'
TEST_DATASET = '../sub1_data/fold1/dailydialog_qa_test_without_context.csv'
# TEST_DATASET = '../sub1_data/fold3/iemocap_qa_test_with_context.csv'

TRANSFORMER_CACHE = '/data/bashwani/Workspace/.cache'

# emotion_model_name = 'monologg/bert-base-cased-goemotions-original'
emotion_model_name= 'bert-base-cased'

EMOTION_CONFIG = BertConfig.from_pretrained(emotion_model_name, cache_dir=TRANSFORMER_CACHE)
EMOTION_TOKENIZER = BertTokenizer.from_pretrained(emotion_model_name, config=EMOTION_CONFIG, cache_dir=TRANSFORMER_CACHE)

cause_model_name = 'mrm8488/spanbert-finetuned-squadv2'
CAUSE_CONFIG = BertConfig.from_pretrained(cause_model_name, cache_dir=TRANSFORMER_CACHE)
CAUSE_TOKENIZER = BertTokenizer.from_pretrained(cause_model_name, config=CAUSE_CONFIG, cache_dir=TRANSFORMER_CACHE)

emotion_mapping = {
    'happiness': 0, 
    'surprise': 1,
    'anger': 2,
    'sadness': 3,
    'disgust': 4,
    'fear': 5,
    'excited': 6
}

n_classes = len(emotion_mapping)