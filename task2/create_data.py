import pandas as pd
import numpy as np
from tqdm import tqdm
def create_data(path, dest):
    df = pd.read_csv(path)
    df_new = pd.DataFrame(columns=['Ut', 'Ui', 'context', 'emotion','labels', 'id'])
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating Dataset"):
        text = df.iloc[idx]['text'].split(' <SEP> ')

        
        df_new.loc[idx, 'emotion'] = text[0]
        df_new.loc[idx, 'Ut'] = text[1]
        df_new.loc[idx, 'Ui'] = text[2]
        df_new.loc[idx, 'context'] = text[3]
        df_new.loc[idx, 'labels'] = df.loc[idx, 'labels']
        df_new.loc[idx, 'id'] = df.loc[idx, 'id']
    
    df_new.to_csv('./data/'+dest, index=False)

if __name__ == '__main__':
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold1/dailydialog_classification_train_with_context.csv', 'fold1/dailydialog_classification_train_with_context.csv')
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold1/dailydialog_classification_valid_with_context.csv', 'fold1/dailydialog_classification_valid_with_context.csv')
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold1/dailydialog_classification_test_with_context.csv', 'fold1/dailydialog_classification_test_with_context.csv')

    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold2/dailydialog_classification_train_with_context.csv', 'fold2/dailydialog_classification_train_with_context.csv')
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold2/dailydialog_classification_valid_with_context.csv', 'fold2/dailydialog_classification_valid_with_context.csv')
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold2/dailydialog_classification_test_with_context.csv', 'fold2/dailydialog_classification_test_with_context.csv')
    
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold3/dailydialog_classification_train_with_context.csv', 'fold3/dailydialog_classification_train_with_context.csv')
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold3/dailydialog_classification_valid_with_context.csv', 'fold3/dailydialog_classification_valid_with_context.csv')
    create_data('/data/bashwani/Workspace/RECCON/data/subtask2/fold3/dailydialog_classification_test_with_context.csv', 'fold3/dailydialog_classification_test_with_context.csv')




