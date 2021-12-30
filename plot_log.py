import argparse
import json
import os

import matplotlib.pyplot as plt



def get_log(log_path):
    with open(log_path, 'r') as f:
        log = json.load(f)
        
    epochs, loss, acc, f1 = [], [], [], []
    for history in log['log_history']:
        if history['epoch'] == int(history['epoch']) and 'eval_loss' in history:
            epochs.append(history['epoch'])
            loss.append(history['eval_loss'])
            acc.append(history['eval_accuracy'])
            f1.append(history['eval_f1'])

    return {
        'epoch': epochs, 
        'loss': loss,
        'acc': acc, 
        'f1': f1
    }


def plot_log(mode):
    
    x = base_log['epoch']
    plt.plot(x, base_log[mode], color='red', label='roberta_base')
    plt.plot(x, pre_ce_log[mode], color='blue', label='roberta_pretrained_ce')
    plt.plot(x, pre_focal_log[mode], color='green', label='roberta_pretrained_focal')
    plt.title(f'validation {mode}')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.savefig(f'{mode}.png')
    plt.close()
    

if __name__ == '__main__':
    
    base_ce_log_path = 'qqp_roberta-base_ce/trainer_state.json'
    pre_ce_log_path = 'qqp_roberta-base_pretrained_e10_ce/trainer_state.json'
    pre_focal_log_path = 'qqp_roberta-base_pretrained_e10_focal/trainer_state.json'
    
    base_log = get_log(base_ce_log_path)
    pre_ce_log = get_log(pre_ce_log_path)
    pre_focal_log = get_log(pre_focal_log_path)
    
    plot_log('loss')
    plot_log('acc')
    plot_log('f1')
    