# checking the max len of utterance after tokenization
import pandas as pd
from transformers import (
    BertTokenizer,
    BertModel
)

import seaborn as sns
import matplotlib.pyplot as plt

def utterance_length(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    utterances = df.utterance
    ulen = []
    lst = []
    for idx, u in enumerate(utterances):
        iids = tokenizer.encode(u, add_special_tokens=False)
        ulen.append({'l': len(iids), 'u': u, 'iids': iids})
        lst.append(len(iids))

    ls = sorted(ulen, key=lambda x: x['l'])
    lst = sorted(lst)
    print(lst)
    print(max(ulen, key=lambda x: x['l']))
    print()

# dont need to call countplot, perform all visualizations on jupyterlab
def countplot(df, name):
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    
    sns_plot = sns.countplot('emotion', data=df)
    fig = sns_plot.get_figure()
    fig.savefig('figures/' + name+'_countplot_emotion.png')

def data_insights(df, name):
    print(df['emotion'].value_counts())
    
if __name__ == '__main__':
    df_train = pd.read_csv('./transformed_data/dd_train.csv')
    df_valid = pd.read_csv('./transformed_data/dd_valid.csv')
    df_test = pd.read_csv('./transformed_data/dd_test.csv')
    
    # print('Train data')
    # utterance_length(df_train)
    # print("*"*50)
    # print("Valid data")
    # utterance_length(df_valid)
    # print("*"*50)
    # print("Test data")
    # utterance_length(df_test)

    data_insights(df_train, 'train')
    data_insights(df_valid, 'valid')
    data_insights(df_test, 'test')



