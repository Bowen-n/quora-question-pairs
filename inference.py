# @Time: 2021/12/17 15:51
# @Author: Bolun Wu

import argparse
import json
import os

import numpy as np
import torch
import tqdm
from datasets import load_dataset, load_metric
from transformers import (RobertaForSequenceClassification,
                          RobertaTokenizerFast, Trainer, TrainingArguments)
from transformers.data.data_collator import DataCollatorWithPadding


torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=('train', 'valid', 'test', 'all'), default='train')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='inference')
args = parser.parse_args()

# * Dataset
data_files = {'train': 'data/train.txt',
              'validation': 'data/valid.txt',
              'test': 'data/test.txt'}
dataset = load_dataset('csv',
                       data_files=data_files,
                       column_names=['text_a', 'text_b', 'labels'],
                       delimiter='\t',
                       quoting=3)
# filter None sentence
dataset = dataset.filter(lambda example: example['text_a'] and example['text_b'])

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def tokenize_function(example):
    return tokenizer(example['text_a'], example['text_b'], padding='longest')

tokenized_datasets = dataset.map(tokenize_function, batched=True)
full_train_dataset = tokenized_datasets['train']
full_eval_dataset = tokenized_datasets['validation']
full_test_dataset = tokenized_datasets['test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')


def compute_metrics(eval_pred):
    metric1 = load_metric("metrics/accuracy.py")
    metric2 = load_metric("metrics/f1.py")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)['accuracy']
    f1 = metric2.compute(predictions=predictions, references=labels, average='macro')['f1']
    return {"accuracy": accuracy, "f1": f1}


def inference_main(dataset, type_):
    
    model = RobertaForSequenceClassification.from_pretrained('pretrain_roberta_e10')
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    training_args = TrainingArguments(per_device_eval_batch_size=256,
                                      eval_accumulation_steps=1,
                                      dataloader_num_workers=24,
                                      output_dir=args.output_dir,
                                      fp16=True, seed=2021)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics)
    
    preds, label_ids, metrics = trainer.predict(dataset)
    np.savetxt(os.path.join(args.output_dir, f'{type_}_predictions'), preds)
    np.savetxt(os.path.join(args.output_dir, f'{type_}_label_ids'), label_ids)
    with open(os.path.join(args.output_dir, f'{type_}_metrics'), 'w') as f:
        f.write(json.dumps(metrics, indent=1))


if args.mode == 'train':
    inference_main(full_train_dataset, 'train')
elif args.mode == 'valid':
    inference_main(full_eval_dataset, 'valid')
elif args.mode == 'test':
    inference_main(full_test_dataset, 'test')
elif args.mode == 'all':
    inference_main(full_train_dataset, 'train')
    inference_main(full_eval_dataset, 'valid')
    inference_main(full_test_dataset, 'test')

