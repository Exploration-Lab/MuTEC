WANDB = False
EPOCH=5
TRAIN_BATCH_SIZE=14
VALID_BATCH_SIZE=14
ACCUMULATION_STEPS=4
MAX_LENGTH=512
LEARNING_RATE = 4e-5
BETA = 1
TRAIN_DATASET = './data/fold1/dailydialog_classification_train_with_context.csv'
VALID_DATASET = './data/fold1/dailydialog_classification_valid_with_context.csv'
TEST_DATASET = './data/fold1/dailydialog_classification_test_with_context.csv'
# TEST_DATASET = './data/fold1/iemocap_classification_test_with_context.csv'

TRANSFORMER_CACHE = '/data/bashwani/Workspace/.cache'

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