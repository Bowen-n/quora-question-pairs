# @Time: 2021/12/12 19:47
# @Author: Bolun Wu

import argparse
import os

from datasets import load_dataset
from transformers import (RobertaForMaskedLM, RobertaTokenizerFast, Trainer,
                          TrainingArguments)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_utils import SchedulerType


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='continue pretraining from a checkpoint')

parser.add_argument('--save_total_limit', type=int, default=-1)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--warmup_ratio', type=float, default=0.05)
parser.add_argument('--use_fp16', action='store_true', default=False)
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--output_dir', type=str)

parser.add_argument('--seed', type=int, default=2021)
args = parser.parse_args()
print(args)

pretrained_name = 'roberta-base'

dataset = load_dataset('csv',
                       data_files=['data/train.txt', 'data/valid.txt', 'data/test.txt'],
                       column_names=['text_a', 'text_b', 'label'],
                       delimiter='\t',
                       quoting=3)['train']

# filter None sentence
dataset = dataset.filter(lambda example: example['text_a'] and example['text_b'])

tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_name)

def tokenize_function(example):
    return tokenizer(example['text_a'], example['text_b'], padding='longest')

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['label'])

# MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

model = RobertaForMaskedLM.from_pretrained(pretrained_name)

training_args = TrainingArguments(
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    lr_scheduler_type=SchedulerType.LINEAR,
    warmup_ratio=args.warmup_ratio,
    output_dir=args.output_dir,
    save_strategy='epoch',
    dataloader_num_workers=args.num_workers,
    save_total_limit=args.save_total_limit,
    logging_steps=args.logging_steps,
    logging_first_step=True,
    fp16=args.use_fp16,
    seed=args.seed
)

trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=tokenized_datasets)

trainer.train(resume_from_checkpoint=args.resume_checkpoint_path)
trainer.save_model(args.output_dir)
trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))

tokenizer.save_vocabulary(training_args.output_dir, 'vocab')

