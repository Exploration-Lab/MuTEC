import json
import pandas as pd


def dis(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def dialog_utterance(path, df_name):
    df = pd.DataFrame(columns=['dialog_id', 'utterance_list'])
    with open(path, encoding='utf-8') as f:
        dialog_dict = json.load(f)

    for idx, (dialog, values) in enumerate(dialog_dict.items()):
        df.loc[idx, 'dialog_id'] = dialog
        utterance_list = []
        for entry in values[0]:
            utterance_list.append(entry['utterance'])
        
        df.loc[idx, 'utterance_list'] =  utterance_list

    df.to_csv(df_name, index=False)

def transform_data(dialog_dict, df_name, dialog):
    dialog_utterance = pd.read_csv(dialog)

    data = pd.DataFrame(columns=['idx', 'emotion', 'utterance', 'cause_utterance', 'cause_span', 'history'])    
    overall_idx = 0
    for diag_idx, (dialog, values) in enumerate(dialog_dict.items()): 
        # extract it for one dialog
        current_idx = 1
        history = ''
        for entry in values[0]:
            data.loc[overall_idx, 'idx'] = current_idx
            data.loc[overall_idx, 'emotion'] = entry['emotion']
            data.loc[overall_idx, 'utterance'] = entry['utterance']
            history += entry['utterance'] + ' '
            data.loc[overall_idx, 'history'] = history
            
            if entry['emotion'] == 'neutral':
                data.loc[overall_idx, 'cause_utterance'] = 'null'
                data.loc[overall_idx, 'cause_span'] = 'null'
                current_idx += 1
                overall_idx += 1
                
            else:
                cause_index = entry['expanded emotion cause evidence'][0]
                if cause_index != 'b':
                    data.loc[overall_idx, 'cause_utterance'] = eval(dialog_utterance.loc[diag_idx, 'utterance_list'])[cause_index-1]
                    data.loc[overall_idx, 'cause_span'] = entry['expanded emotion cause span'][0]
                    current_idx += 1
                    overall_idx += 1

                if(len(entry['expanded emotion cause evidence']) > 1):
                    for i, (cause_idx, span) in enumerate(zip(entry['expanded emotion cause evidence'][1:], entry['expanded emotion cause span'][1:])):
                        # data.loc[overall_idx, :] = data.loc[overall_idx-1, :].copy()
                        if cause_idx != 'b':
                            data.loc[overall_idx, 'emotion'] = entry['emotion']
                            data.loc[overall_idx, 'utterance'] = entry['utterance']
                            data.loc[overall_idx, 'history'] = history
                            data.loc[overall_idx, 'idx'] = current_idx
                            # print(dialog)

                            data.loc[overall_idx, 'cause_utterance'] = eval(dialog_utterance.loc[diag_idx, 'utterance_list'])[cause_idx-1]
                            # print(cause_idx, '\t', eval(dialog_utterance.loc[diag_idx, 'utterance_list'])[cause_idx-1])
                            data.loc[overall_idx, 'cause_span'] = span            
                            current_idx += 1
                            overall_idx += 1

                elif(len(entry['expanded emotion cause evidence']) == 1 and cause_index == 'b'):
                    data.loc[overall_idx, 'cause_utterance'] = 'null'
                    data.loc[overall_idx, 'cause_span'] = 'null'
                    current_idx += 1
                    overall_idx += 1
    
    data.to_csv(df_name, index=False)

def call_transform(path, df_name, dialog):
    with open(path, encoding='utf-8') as f:
        dialog_dict = json.load(f)

    transform_data(dialog_dict, df_name, dialog)

if __name__ == '__main__':
    '''
    dialog_utterance('../data/original_annotation/dailydialog_train.json', 'transformed_data/dialog_utterances_train.csv')
    dialog_utterance('../data/original_annotation/dailydialog_valid.json', 'transformed_data/dialog_utterances_valid.csv')
    dialog_utterance('../data/original_annotation/dailydialog_test.json', 'transformed_data/dialog_utterances_test.csv')
    '''
    
    call_transform('../data/original_annotation/dailydialog_train.json', 'transformed_data/dd_train.csv', './transformed_data/dialog_utterances_train.csv')
    call_transform('../data/original_annotation/dailydialog_valid.json', 'transformed_data/dd_valid.csv', './transformed_data/dialog_utterances_valid.csv')
    call_transform('../data/original_annotation/dailydialog_test.json', 'transformed_data/dd_test.csv', './transformed_data/dialog_utterances_test.csv')

    # call_transform('../data/original_annotation/iemocap_test.json', 'iemocap_test.csv')
