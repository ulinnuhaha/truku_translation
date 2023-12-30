#!/usr/bin/env python
# coding: utf-8
# import the libraries
import os
import torch
import random
import evaluate
import numpy as np
from dataclasses import dataclass
from time import perf_counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
# use argparse to let the user provides values for variables at runtime
def DataTrainingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', 
        type=str, required=True, help='Load a LLM as model checkpoint for translation')
    parser.add_argument('--train_file', 
        type=str, required=True, help='Data training file')
    parser.add_argument('--eval_file', 
        type=str, required=True, help='Data evaluation file')
    parser.add_argument('--cache_dir', 
        type=str, required=True, help='Directory for saving the pre-trained translation model')
    parser.add_argument('--trans_direction', 
        type=str, required=True, help='Translation direction, tr2ch or ch2tr')
        # tr2ch: Truku to Chinese & ch2tr: Chinese to Truku
    args = parser.parse_args()
    return args

#create the configuration class
@dataclass
class Config:
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    max_source_length: int = 128 # the maximum length in number of tokens for tokenizing the input sentence
    max_target_length: int = 128 # the maximum length in number of tokens for tokenizing the target sentence

    lr: float = 0.0005
    weight_decay: float = 0.01
    epochs: int = 20
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set random seed to ensure that results are reproducible
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

def main():
    data_train_args=DataTrainingArguments() #call the arguments
    config = Config()
    #Load the training dataset from tsv files
    data_file = {}
    data_file["train"] = data_train_args.train_file
    data_file["val"] = data_train_args.eval_file
    
    dataset_dict = load_dataset(
        "csv",
        delimiter="\t",
        column_names=["source_lang", "target_lang"],
        data_files=data_file
    )
    print(dataset_dict)
    #Load the evaluation metrics
    rouge_score = evaluate.load("rouge")
    bleu_score = evaluate.load("bleu")
    sacrebleu_score = evaluate.load("sacrebleu")
    chrf_score = evaluate.load("chrf")
    
    # Load the tokenizer and the pre-trained mT5 model to perform fine-tuning translation
    tokenizer = AutoTokenizer.from_pretrained(data_train_args.model_checkpoint)
    model_name = data_train_args.model_checkpoint.split("/")[-1] #the name of pre-trained model
    
    fine_tuned_model_checkpoint = os.path.join(
        data_train_args.cache_dir,
        f"{model_name}_{data_train_args.trans_direction}"
    )
    
    if os.path.isdir(fine_tuned_model_checkpoint): #load the fine-tuned translation model if available
        do_train = False
        model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_model_checkpoint, cache_dir=data_train_args.cache_dir)
    else: #load the checkpoint model from pre-trained mT5 model as initial checkpoint for translation model
        do_train = True
        model = AutoModelForSeq2SeqLM.from_pretrained(data_train_args.model_checkpoint, cache_dir=data_train_args.cache_dir)
    
    print("number of parameters:", model.num_parameters())
    def batch_tokenize_fn(examples):
        """
        Generate the input_ids and labels field for dataset dict of training data.
        """
        sources = examples["source_lang"]
        targets = examples["target_lang"]
        # tokenizing the input sentences
        model_inputs = tokenizer(sources, max_length=config.max_source_length, truncation=True)
    
        # tokenizing the target sentences
        # tokenized ids of the target are stored as the labels field
        labels = tokenizer(targets, max_length=config.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    #tokenizing the input and target sentences     
    dataset_dict_tokenized = dataset_dict.map(
        batch_tokenize_fn,
        batched=True,
        remove_columns=dataset_dict["train"].column_names
    )
    
    output_dir = os.path.join(data_train_args.cache_dir, f"{model_name}_{data_train_args.trans_direction}") # where the pre-trained translation model is saved
    
    #The training arguments for the training session
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
        # https://discuss.huggingface.co/t/mixed-precision-for-bfloat16-pretrained-models/5315
        fp16=False
    )
    # evalution metrics computation
    def compute_metrics(eval_pred):
        """
        Compute rouge, chrF, and bleu metrics for seq2seq model generated prediction.
        
        tip: we can run trainer.predict on our eval dataset to see what a sample
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
        result["sacrebleu"] = score["score"] #The higher the value, the better the translations
        result["chrf"] = chrf["score"]
        return {k: round(v, 4) for k, v in result.items()}
        
    # Data collator used for seq2seq model
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
    
    # perform the training process
    if trainer.args.do_train:
        t1_start = perf_counter()
        train_output = trainer.train()
        t1_stop = perf_counter()
        print("Training elapsed time:", t1_stop - t1_start)
    
        # saving the pre-trained model
        trainer.save_model(fine_tuned_model_checkpoint)
    
    trainer.evaluate()
if __name__ == "__main__":
    main()
