import json
import pandas as pd
from tqdm import tqdm 

def process_question(question):
    emotion = question.split("What is the causal span from context that is relevant to the target utterance's emotion ")[1][:-2]

    utterance = question.split(" The evidence utterance is")[0][24:]

    # fold 2, fold 3 does not contain cause utterance in the question
    cause_utterance = question.split(" The evidence utterance is ")[1].split(" What is the causal span from context that is relevant to the target utterance's emotion")[0]
    return utterance, cause_utterance, emotion

def transform_data(df_name, path, dataset_name):
    with open(path, encoding='utf-8') as f:
        dialog_dict = json.load(f)
    
    data = pd.DataFrame(columns=['id', 'emotion', 'utterance', 'cause_utterance', 'cause_span', 'context'])    
    
    for idx, dialog in tqdm(enumerate(dialog_dict), total=len(dialog_dict), desc='Creating {} dataset'.format(dataset_name)):
        data.loc[idx, 'context'] = dialog['context']
        
        for elem in dialog['qas']:
            data.loc[idx, 'id'] = elem['id']
            data.loc[idx, 'utterance'], data.loc[idx, 'cause_utterance'], data.loc[idx, 'emotion'] = process_question(elem['question'])
            data.loc[idx, 'cause_span'] = elem['answers'][0]['text']

    m = {
        'happiness': 'happiness',
        'happines': 'happiness',
        'surprise': 'surprise',
        'anger': 'anger',
        'sadness': 'sadness',
        'disgust': 'disgust',
        'fear': 'fear',
        'excited': 'excited',
        'happy': 'happiness',
        'sad': 'sadness',
        'surprised': 'surprise',
        'angry': 'anger',
        'frustrated': 'frustrated',
    }

    data['emotion'] = data['emotion'].apply(lambda x: m[x])
    data.to_csv(df_name, index=False)

def create_qa(fold):
    train_path = '../data/subtask1/' + fold + '/dailydialog_qa_train_with_context.json'
    valid_path = '../data/subtask1/' + fold + '/dailydialog_qa_valid_with_context.json'
    test_dd_path = '../data/subtask1/' + fold + '/dailydialog_qa_test_with_context.json'
    test_iemocap_path = '../data/subtask1/' + fold + '/iemocap_qa_test_with_context.json'
    
    transform_data('transformed_data/' + fold + '/dd_qa_train.csv', train_path, 'Train')
    transform_data('transformed_data/' + fold + '/dd_qa_valid.csv', valid_path, 'Valid') 
    transform_data('transformed_data/' + fold + '/dd_qa_test.csv', test_dd_path, 'Test dd') 
    transform_data('transformed_data/' + fold + '/iemocap_qa_test.csv', test_iemocap_path, 'Test iemocap') 

if __name__ == '__main__':

    create_qa('fold1') 
    print("Fold1 data created. . . .")

    
