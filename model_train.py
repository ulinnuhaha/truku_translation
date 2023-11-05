
import os
import torch
import random
import pandas as pd
import evaluate
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from time import perf_counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
print(torch.cuda.is_available())

import gc
gc.collect()
torch.cuda.empty_cache()

seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
class Config:
    cache_dir: str = "./translation"
    data_dir: str = "./dataset"
    lang: str = "tr_ch"
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    max_source_length: int = 128
    max_target_length: int = 128

    lr: float = 0.0005
    weight_decay: float = 0.01
    epochs: int = 20
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_checkpoint: str = "facebook/nllb-200-distilled-600M"

    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)


config = Config()

data_file = {}
for split in ["train", "val", "test"]:
    output_path = os.path.join(config.data_dir, f"{split}.tsv")
    data_file[split] = [output_path]

dataset_dict = load_dataset(
    "csv",
    delimiter="\t",
    column_names=["source_lang", "target_lang"],
    data_files=data_file
)
print(dataset_dict)

rouge_score = evaluate.load("rouge")
bleu_score = evaluate.load("bleu")
sacrebleu_score = evaluate.load("sacrebleu")
chrf_score = evaluate.load("chrf")
bert_score= evaluate.load("bertscore")
print(dataset_dict['train'][0])

# # Fine Tuning Pretrained Model
tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)

model_name = config.model_checkpoint.split("/")[-1]

fine_tuned_model_checkpoint = os.path.join(
    config.cache_dir,
    f"{model_name}_{config.lang}"
)

if os.path.isdir(fine_tuned_model_checkpoint):
    do_train = False
    model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_checkpoint, cache_dir=config.cache_dir)
else:
    do_train = True
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint, cache_dir=config.cache_dir)

print("number of parameters:", model.num_parameters())
def batch_tokenize_fn(examples):
    """
    Generate the input_ids and labels field for huggingface dataset/dataset dict.

    Truncation is enabled where we cap the sentence to the max length. Padding will be done later
    in a data collator, so we pad examples to the longest length within a mini-batch and not
    the whole dataset.
    """
    sources = examples["source_lang"]
    targets = examples["target_lang"]
    model_inputs = tokenizer(sources, max_length=config.max_source_length, truncation=True)

    # setup the tokenizer for targets,
    # huggingface expects the target tokenized ids to be stored in the labels field
    # note, newer version of tokenizer supports a text_target argument, where we can create
    # source and target sentences in one go
   # with tokenizer.as_target_tokenizer():
    labels = tokenizer(targets, max_length=config.max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
dataset_dict_tokenized = dataset_dict.map(
    batch_tokenize_fn,
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)
print(dataset_dict["train"][121])
print(dataset_dict["train"][278])

# ## Data collator used for seq2seq model

model_name = config.model_checkpoint.split("/")[-1]
output_dir = os.path.join(config.cache_dir, f"{model_name}_{config.lang}")

args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    learning_rate=config.lr,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    weight_decay=config.weight_decay,
    save_total_limit=2,
    num_train_epochs=config.epochs,
    predict_with_generate=True,
    load_best_model_at_end=True,
    greater_is_better=False, #lower score better result of the main metric
    metric_for_best_model="eval_loss", #set metrics as the main parameter
    gradient_accumulation_steps=8,
    do_train=do_train,
    # careful when attempting to train t5 models on fp16 mixed precision,
    # the model was trained on bfloat16 mixed precision, and mixing different mixed precision
    # type might result in nan loss
    # https://discuss.huggingface.co/t/mixed-precision-for-bfloat16-pretrained-models/5315
    fp16=False
)

def compute_metrics(eval_pred):
    """
    Compute rouge and bleu metrics for seq2seq model generated prediction.
    
    tip: we can run trainer.predict on our eval/test dataset to see what a sample
    eval_pred object would look like when implementing custom compute metrics function
    """
    predictions, labels = eval_pred
    # Decode generated summaries, which is in ids into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode labels, a.k.a. reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1", "rouge2", "rougeL"] #A ROUGE score close to zero indicates poor similarity between candidate and references. A ROUGE score close to one indicates strong similarity between candidate and references
    )
    score = sacrebleu_score.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    chrf=chrf_score.compute(predictions=decoded_preds, references=decoded_labels)
    #berts = bert_score.compute(predictions=decoded_preds, references=decoded_labels,  model_type="bert-base-chinese")
    #result["bert_score"]= np.mean(berts['f1'])
    result["sacrebleu"] = score["score"] #The higher the value, the better the translations
    result["chrf"] = chrf["score"]
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=dataset_dict_tokenized["train"],
    eval_dataset=dataset_dict_tokenized["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(2, 0.0)] #early_stopping_patience =2, early_stopping_threshold =0
)

# should take around 4117.78 seconds on a single V100 GPU
if trainer.args.do_train:
    t1_start = perf_counter()
    train_output = trainer.train()
    t1_stop = perf_counter()
    print("Training elapsed time:", t1_stop - t1_start)

    # saving the model which allows us to leverage
    # .from_pretrained(model_path)
    trainer.save_model(fine_tuned_model_checkpoint)

trainer.evaluate()
