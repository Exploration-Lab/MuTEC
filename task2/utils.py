import config
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os
import seaborn as sns


m = {
  'happines': 'happiness',
  'happy': 'happiness',
  'angry': 'anger',
  'sad': 'sadness',
  'surprised': 'surprise',
  'frustrated': 'anger',
}


def plot_loss(train_loss, eval_loss, name):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(1, len(train_loss)+1), train_loss, label='train_loss')
    ax.plot(range(1, len(eval_loss)+1), eval_loss, label='eval_loss')
    
    plt.xticks(range(1, len(train_loss)+1))
    plt.legend()
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.title("Loss Progression")
    plt.savefig(name+'_fig.png')
    fig.clear()
    

def plot_grad_flow(named_parameters, path, step):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class / vanilla training loop after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters(), 'output_path', step)" to visualize the gradient flow
    and save a plot in the output path as a series of .png images.
    Adapted from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.3, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.3, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=1e-8, top=20)  # zoom in on the lower gradient regions
    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.savefig(os.path.join(
        path, f"{step:04d}.png"), bbox_inches='tight', dpi=200)
    fig.clear()


def preprocess(df):
  df['emotion'] = df['emotion'].replace(m)
  return df


def get_weights(df):
  label_weights = [0]*2
  em_weights = [0]*6
  em = df['emotion'].value_counts().reset_index().values.tolist()
  lm = df['labels'].value_counts().reset_index().values.tolist()
  
  
  for idx, (e, c) in enumerate(em):
    em_weights[config.emotion_mapping[e]] = c
  
  for idx, (e, c) in enumerate(lm):
    label_weights[e] = c

  return  label_weights, em_weights

def plot_len(x, name):
  fig = plt.figure()
  sns.distplot(x, hist=True, kde=True, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
  plt.xlabel("Number of tokens")
  plt.ylabel("Density")
  plt.title("Distribution plot of tokens")
  plt.savefig(name+'_fig.png')
  fig.clear()

def calculate_pos_per(outputs, targets, em_outputs, em_targets):
  tar_pos_count = sum(targets)
  tar_neg_count = len(targets)-tar_pos_count 
  out_pos_count = sum(outputs)
  out_neg_count = len(outputs)-out_pos_count
  mp = {y: 0 for x,y in config.emotion_mapping.items()}
  
  for x, y, w, z in zip(em_outputs, em_targets, outputs, targets):
    if x == y and w == z:
      mp[x] = mp.get(x, 0)+1


  print("Ratio Targets:", {k: v/len([x for x in em_targets if x==k]) for k, v in mp.items()})    
  print("Ratio Predicted", {k: v/(1 if len([x for x in em_outputs if x==k])==0 else len([x for x in em_outputs if x==k])) for k, v in mp.items()})    



class AverageMeter(object):
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

