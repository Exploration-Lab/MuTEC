WANDB = False
EPOCH=1
TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=4
ACCUMULATION_STEPS=4
MAX_LENGTH=512
LEARNING_RATE = 4e-5
BETA = 1
TRAIN_DATASET = '../balanced/subtask2/fold1/dailydialog_classification_train_with_context.csv'
VALID_DATASET = '../balanced/subtask2/fold1/dailydialog_classification_valid_with_context.csv'
TEST_DATASET = '../balanced/subtask2/fold1/dailydialog_classification_test_with_context.csv'
# TEST_DATASET = './data/fold1/iemocap_classification_test_with_context.csv'

TRANSFORMER_CACHE = '/data/bashwani/Workspace/.cache'
# PATH = 'best_model_wocontext/'  
PATH = 'best_model/'
# PATH = 'best_model_with_context/'

emotion_mapping = {
    'happiness': 0, 
    'surprise': 1,
    'anger': 2,
    'sadness': 3,
    'disgust': 4,
    'fear': 5,
}

n_classes = len(emotion_mapping)