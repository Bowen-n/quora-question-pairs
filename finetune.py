# @Time: 2021/12/14 19:51
# @Author: Bolun Wu

import argparse
import json
import os

import numpy as np
import tqdm
from datasets import load_dataset, load_metric
from transformers import (RobertaForSequenceClassification,
                          RobertaTokenizerFast, Trainer, TrainingArguments)
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_utils import SchedulerType

from model import RobertaForSequenceClassification_FocalLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument('--pre_name', type=str, default='roberta-base')
parser.add_argument('--pretrained_model_dir', type=str)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--loss_type', type=str, choices=('focal', 'ce'), default='focal')
parser.add_argument('--focal_loss_alpha', type=float, nargs='+', default=0.25)
parser.add_argument('--focal_loss_gamma', type=float, default=2)

parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--warmup_ratio', type=float, default=0.05)
parser.add_argument('--use_fp16', action='store_true', default=False)
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--output_dir', type=str)

parser.add_argument('--seed', type=int, default=2021)
args = parser.parse_args()
print(args)

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

tokenizer = RobertaTokenizerFast.from_pretrained(args.pre_name)

def tokenize_function(example):
    return tokenizer(example['text_a'], example['text_b'], padding='longest')

tokenized_datasets = dataset.map(tokenize_function, batched=True)
full_train_dataset = tokenized_datasets['train']
full_eval_dataset = tokenized_datasets['validation']
full_test_dataset = tokenized_datasets['test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')


# * Model
if args.loss_type == 'ce':
    if args.pretrained_model_dir:
        model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_dir, num_labels=2)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.pre_name, num_labels=2)

elif args.loss_type == 'focal':
    if args.pretrained_model_dir:
        model = RobertaForSequenceClassification_FocalLoss.from_pretrained(args.pretrained_model_dir, num_labels=2)
    else:
        model = RobertaForSequenceClassification_FocalLoss.from_pretrained(args.pre_name, num_labels=2)
    model.config_loss(args.focal_loss_alpha, args.focal_loss_gamma)
        

# * Train spec
training_args = TrainingArguments(
    num_train_epochs=args.num_epochs,
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    eval_accumulation_steps=1,
    learning_rate=args.learning_rate,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=args.warmup_ratio,
    output_dir=args.output_dir,
    save_strategy='epoch',
    dataloader_num_workers=args.num_workers,
    save_total_limit=1,
    logging_steps=args.logging_steps,
    logging_first_step=True,
    fp16=args.use_fp16,
    seed=args.seed
)


def compute_metrics(eval_pred):
    metric1 = load_metric("metrics/accuracy.py")
    metric2 = load_metric("metrics/f1.py")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric1.compute(predictions=predictions, references=labels)['accuracy']
    f1 = metric2.compute(predictions=predictions, references=labels, average='macro')['f1']
    return {"accuracy": accuracy, "f1": f1}


trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=full_train_dataset,
                  eval_dataset=full_eval_dataset,
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

# train
trainer.train()
trainer.save_model(args.output_dir)
trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))

# test
test_predictions, test_label_ids, test_metrics = trainer.predict(full_test_dataset)
np.savetxt(os.path.join(args.output_dir, 'test_predictions'), test_predictions)
np.savetxt(os.path.join(args.output_dir, 'test_label_ids'), test_label_ids)
with open(os.path.join(args.output_dir, 'test_metrics'), 'w') as f:
    f.write(json.dumps(test_metrics, indent=1))

# save vocab
tokenizer.save_vocabulary(training_args.output_dir, 'vocab')

