import argparse
import json
import os

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

base_dir = 'inference_base_ce'
pre_ce_dir = 'inference_pre_ce'
pre_focal_dir = 'inference_pre_focal'

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='stacking')
parser.add_argument('--model_path', type=str)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


def load_data_label(mode):
    
    base = np.loadtxt(os.path.join(base_dir, f'{mode}_predictions'))
    base = np.array(F.softmax(torch.tensor(base), dim=1))[:, 0].reshape(-1, 1)
    
    ce = np.loadtxt(os.path.join(pre_ce_dir, f'{mode}_predictions'))
    ce = np.array(F.softmax(torch.tensor(ce), dim=1))[:, 0].reshape(-1, 1)

    focal = np.loadtxt(os.path.join(pre_focal_dir, f'{mode}_predictions'))
    focal = np.array(F.softmax(torch.tensor(focal), dim=1))[:, 0].reshape(-1, 1)

    label = np.loadtxt(os.path.join(pre_ce_dir, f'{mode}_label_ids'))
    return np.concatenate([base, ce, focal], axis=1), label


# load dataset
if not args.test:
    train_data, train_label = load_data_label('train')
    valid_data, valid_label = load_data_label('valid')
    print(f'Train {train_data.shape} Valid {valid_data.shape}')

test_data, test_label = load_data_label('test')
print(f'Test {test_data.shape}')


# train linear model
if not args.test:
    best_model = None
    best_valid_f1 = 0
    for c in tqdm.tqdm([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], desc='grid search'):
        model = LogisticRegression(class_weight={0: 0.37, 1: 0.63},
                                   C=c, n_jobs=-1,
                                   max_iter=200,
                                   random_state=2021)
        model.fit(train_data, train_label)
        valid_preds = model.predict(valid_data)
        valid_f1 = f1_score(valid_label, valid_preds, average='macro')
        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model = model

    model = best_model
    print(f'Final Model Choice: {model}\n')

    # inference
    train_preds, valid_preds, test_preds = \
        model.predict(train_data), model.predict(valid_data), model.predict(test_data)

    # metrics
    train_acc = accuracy_score(train_label, train_preds)
    train_f1 = f1_score(train_label, train_preds, average='macro')

    valid_acc = accuracy_score(valid_label, valid_preds)
    valid_f1 = f1_score(valid_label, valid_preds, average='macro')

    test_acc = accuracy_score(test_label, test_preds)
    test_f1 = f1_score(test_label, test_preds, average='macro')

    print(f'Train acc: {train_acc:.6f} f1: {train_f1:.6f}')
    print(f'Valid acc: {valid_acc:.6f} f1: {valid_f1:.6f}')
    print(f'Test acc: {test_acc:.6f} f1: {test_f1:.6f}')

# load model from disk
else:
    print(f'loading linear model from {args.model_path}')
    model = joblib.load(args.model_path)
    
    test_preds = model.predict(test_data)
    test_acc = accuracy_score(test_label, test_preds)
    test_f1 = f1_score(test_label, test_preds, average='macro')
    print(f'Test acc: {test_acc:.6f} f1: {test_f1:.6f}')

# save metric
test_probas = model.predict_proba(test_data)
os.makedirs(args.output_dir, exist_ok=True)
np.savetxt(os.path.join(args.output_dir, 'test_predictions'), test_preds)
np.savetxt(os.path.join(args.output_dir, 'test_proba'), test_probas)
np.savetxt(os.path.join(args.output_dir, 'test_label_ids'), test_label)
with open(os.path.join(args.output_dir, 'test_metrics'), 'w') as f:
    json.dump({'test_accuracy': test_acc,
               'test_f1': test_f1},
              f, indent=1)
    
# save model
if not args.test:
    joblib.dump(model, os.path.join(args.output_dir, 'linear.model'))

