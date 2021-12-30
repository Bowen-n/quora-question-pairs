# @Time: 2021/12/17 16:30
# @Author: Bolun Wu

import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F


def calculate_gamma(easy, hard):
    return math.log(easy/hard, 9)

def calculate_alpha(label_0, label_1):
    rate_0 = label_0 / (label_0 + label_1)
    rate_1 = label_1 / (label_0 + label_1)
    return rate_1, rate_0


inference_dir = 'inference_ce'

labels = np.loadtxt(os.path.join(inference_dir, 'train_label_ids'))
preds = np.loadtxt(os.path.join(inference_dir, 'train_predictions'))
     
label_count = Counter(labels)
labels, preds = torch.tensor(labels), torch.tensor(preds)
preds = F.softmax(preds, dim=1)

pts = []
for label, pred in zip(labels, preds):
    pts.append(1.0 - pred[int(label)].item())
    
easy_count, hard_count = 0, 0
for pt in pts:
    if pt <= 0.1: easy_count += 1
    elif pt >= 0.9: hard_count += 1


## * get gamma
gamma_ = calculate_gamma(easy_count, hard_count)
print(f'Easy sample: {easy_count}. Hard sample: {hard_count}')
print(f'Estimated gamma: {gamma_:.2f}')

## * get alpha
alpha_0, alpha_1 = calculate_alpha(label_count[0], label_count[1])
print(f'Label 0: {label_count[0]}. Label 1: {label_count[1]}')
print(f'Estimated alpha 0: {alpha_0:.2f}, 1: {alpha_1:.2f}')

sns.set_theme()
g = sns.histplot(pts, bins=10)
g.axes.set_yscale('log')
g.set_title('easy vs hard samples')
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
plt.show()
plt.savefig(os.path.join(inference_dir, 'dist.png'))


with open(os.path.join(inference_dir, 'focal_params.json'), 'w') as f:
    data = {'easy': easy_count,
            'hard': hard_count,
            'label_0': label_count[0],
            'label_1': label_count[1],
            'gamma': gamma_,
            'alpha': [alpha_0, alpha_1]}
    json.dump(data, f, indent=1)
    
