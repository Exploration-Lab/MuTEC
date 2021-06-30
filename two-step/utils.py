import config 
from question_answering_utils import compute_f1
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

m = {
  'happines': 'happiness',
  'happy': 'happiness',
  'angry': 'anger',
  'sad': 'sadness',
  'surprised': 'surprise',
  'frustrated': 'anger',
}

def preprocess(df):
  df['emotion'] = df['emotion'].replace(m)
  return df
  
def to_list(tensor):
    return tensor.detach().cpu().tolist()

def seed_torch(seed=42):
    print("Seed everything. . . .")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def emotion_to_idx(emotion):
    emotion_mapping = config.emotion_mapping
    return emotion_mapping[emotion]

def idx_to_emotion(idx):
    emotion_mapping = config.emotion_mapping
    rev_emotion_mapping = {v: k for k, v in emotion_mapping.items()}
    return rev_emotion_mapping[idx]

def plot_loss(train_loss, eval_loss, name):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(1, len(train_loss)+1), train_loss, label='train_loss')
    ax.plot(range(1, len(eval_loss)+1), eval_loss, label='eval_loss')
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.title("Loss Progression")
    plt.savefig(name+'_fig.png')
    fig.clear()
    

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set


def evaluate_results(text):
    partial_match_scores = []
    lcs_all = []
    impos1, impos2, impos3, impos4 = 0, 0, 0, 0
    pos1, pos2, pos3 = 0, 0, 0
    fscores, squad_fscores = [], [] # f1 for postive (valid) instances
    fscores_all, squad_fscores_all = [], [] # f1 for all instances
    
    for i, key in enumerate(['correct_text', 'similar_text', 'incorrect_text']):
        for item in text[key]:
            if i==0:
                fscores_all.append(1)
                squad_fscores_all.append(1)
                if 'impossible' in item and text[key][item]['predicted'] == '':
                    impos1 += 1
                elif 'span' in item:
                    pos1 += 1
                    fscores.append(1)
                    squad_fscores.append(1)
                    
            elif i==1:
                if 'impossible' in item:
                    impos2 += 1
                    fscores_all.append(1)
                    squad_fscores_all.append(1)
                elif 'span' in item:
                    z = text[key][item]
                    if z['predicted'] != '':
                        longest_match = list(lcs(z['truth'], z['predicted']))[0]
                        lcs_all.append(longest_match)
                        partial_match_scores.append(round(len(longest_match.split())/len(z['truth'].split()), 4))
                        pos2 += 1
                        r = len(longest_match.split())/len(z['truth'].split())
                        p = len(longest_match.split())/len(z['predicted'].split())
                        f = 2*p*r/(p+r)
                        fscores.append(f)
                        squad_fscores.append(compute_f1(z['truth'], z['predicted']))
                        fscores_all.append(f)
                        squad_fscores_all.append(compute_f1(z['truth'], z['predicted']))
                    else:
                        pos3 += 1
                        impos4 += 1
                        fscores.append(0)
                        squad_fscores.append(0)
                        fscores_all.append(0)
                        squad_fscores_all.append(0)                   
                    
            elif i==2:
                fscores_all.append(0)
                squad_fscores_all.append(0)
                if 'impossible' in item:
                    impos3 += 1
                elif 'span' in item:
                    z = text[key][item]
                    if z['predicted'] == '':
                        impos4 += 1
                    pos3 += 1
                    fscores.append(0)
                    squad_fscores.append(0)
                    
    total_pos = pos1 + pos2 + pos3
    imr = impos2/(impos2+impos3)
    imp = impos2/(impos2+impos4)
    imf = 2*imp*imr/(imp+imr)
    
    p1 = 'Postive Samples:'
    p2 = 'Exact Match: {}/{} = {}%'.format(pos1, total_pos, round(100*pos1/total_pos, 2))
    p3 = 'Partial Match: {}/{} = {}%'.format(pos2, total_pos, round(100*pos2/total_pos, 2))
    p4a = 'LCS F1 Score = {}%'.format(round(100*np.mean(fscores), 2))
    p4b = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores), 2))
    p5 = 'No Match: {}/{} = {}%'.format(pos3, total_pos, round(100*pos3/total_pos, 2))
    p6 = '\nNegative Samples:'
    p7 = 'Inv F1 Score = {}%'.format(round(100*imf, 2))
    # p7a = 'Inv Recall: {}/{} = {}%'.format(impos2, impos2+impos3, round(100*imr, 2))
    # p7b = 'Inv Precision: {}/{} = {}%'.format(impos2, impos2+impos4, round(100*imp, 2))
    
    p8 = '\nAll Samples:'
    p9a = 'LCS F1 Score = {}%'.format(round(100*np.mean(fscores_all), 2))
    p9b = 'SQuAD F1 Score = {}%'.format(round(100*np.mean(squad_fscores_all), 2))

    p = '\n'.join([p1, p2, p3, p4a, p4b, p5, p6, p7, p8, p9a, p9b])
    pos_f1 = round(100*np.mean(squad_fscores), 2)
    exact = round(100*pos1/total_pos, 2)
    return p, (exact, pos_f1)


class AverageMeter(object):
	# taken from abhisheks repo
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__