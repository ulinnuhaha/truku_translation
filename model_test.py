#!/usr/bin/env python
# coding: utf-8

import os
import torch
import random
import evaluate
import numpy as np
from dataclasses import dataclass
from datasets import load_dataset
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', 
        type=str, required=True, help='Load a LLM as model checkpoint for translation')
    parser.add_argument('--data_dir', 
        type=str, required=True, help='Directory of the dataset files')
    args = parser.parse_args()
    return args

#create the configuration class
@dataclass
class Config:
    lang: str = "tr_ch"
    seed: int = 42
    max_source_length: int = 128
    max_target_length: int = 128
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

def main():
    data_train_args=DataTestingArguments()
    config = Config()
    # Load the testing data from tsv files
    # Both Truku to Chinese data or Chinese to Truku data
    data_file = {}
    for split in ["test_tru2chi", "test_chi2tru"]:
        output_path = os.path.join(data_train_args.data_dir, f"{split}.tsv")
        data_file[split] = [output_path]
    
    dataset_dict = load_dataset(
        "csv",
        delimiter="\t",
        column_names=["source_lang", "target_lang"],
        data_files=data_file
    )
    print(dataset_dict)
    #Load the evaluation metrics
    rouge_score = evaluate.load("rouge")
    bert_score= evaluate.load("bertscore")
    sacrebleu_score = evaluate.load("sacrebleu")
    chrf_score = evaluate.load("chrf")
    
    # Load the tokenizer and model from pre-trained translation
    tokenizer = AutoTokenizer.from_pretrained(data_train_args.pretrained_model)
    model_name = data_train_args.pretrained_model.split("/")[-1] 
    
    pretrained_model_trans = os.path.join(
        f"{model_name}_{config.lang}"
    )
    
    if os.path.isdir(pretrained_model_trans): #load the pre-trained translation model if available
        model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_trans)
    else:
        print("pre-trained data not found")

    print("number of parameters:", model.num_parameters())
    def batch_tokenize_fn(examples):
        """
        Generate the input_ids and labels field for dataset dict.
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
    # evalution metrics computation
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
        berts = bert_score.compute(predictions=decoded_preds, references=decoded_labels,  model_type="bert-base-chinese")
        result["bert_score"]= np.mean(berts['f1'])
        result["sacrebleu"] = score["score"] #The higher the value, the better the translations
        result["chrf"] = chrf["score"]
        return {k: round(v, 4) for k, v in result.items()}
        
    # Data collator used for seq2seq model
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )
    
    # perform the testing process
    print('+++-----Testing stage for Truku to Chinese-----+++')
    pred_t2c=trainer.predict(dataset_dict_tokenized["test_tru2chi"])
    print(pred_t2c)
    print('+++-----Testing stage for Chinese to Truku-----+++')
    pred_c2t=trainer.predict(dataset_dict_tokenized["test_chi2tru"])
    print(pred_c2t)
if __name__ == "__main__":
    main()
