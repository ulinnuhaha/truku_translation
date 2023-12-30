#!/usr/bin/env python
# coding: utf-8
# import the libraries
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
    NllbTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
# use argparse to let the user provides values for variables at runtime
def DataTestingArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', 
        type=str, required=True, help='Load a NLLB as model checkpoint for translation')
    parser.add_argument('--data_dir', 
        type=str, required=True, help='Directory of the dataset files')
    args = parser.parse_args()
    return args

#create the configuration class
@dataclass
class Config:
    lang: str = "tr_ch_no_prefix"
    seed: int = 42
    max_source_length: int = 128 # the maximum length in number of tokens for tokenizing the input sentence
    max_target_length: int = 128 # the maximum length in number of tokens for tokenizing the target sentence
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data_test_args=DataTestingArguments()
    config = Config()
    # Load the testing data from tsv files
    # Both Truku to Chinese data or Chinese to Truku data
    def get_data(langs):
        data_file = {}
        for split in ["test"]:
            output_path = os.path.join(config.data_dir, langs+f"_{split}.tsv")
            data_file[split] = [output_path]
    
        dataset_dict = load_dataset(
                "csv",
                delimiter="\t",
                #column_names=["source_lang", "target_lang"],
                data_files=data_file
            )
        return dataset_dict
    # load the dataset of sentence pair of (Truku→Chinese)
    dataset_dict_tc=get_data('tc')
    # load the dataset of sentence pair of (Chinese→Truku)
    dataset_dict_ct=get_data('ct')
    print(dataset_dict_tc)
    #Load the evaluation metrics
    rouge_score = evaluate.load("rouge")
    bert_score= evaluate.load("bertscore")
    sacrebleu_score = evaluate.load("sacrebleu")
    chrf_score = evaluate.load("chrf")
    
    # load the fine-tuned translation model
    model_name = data_test_args.model_name_or_path 
        
    if os.path.isdir(model_name): #load the fine-tuned translation model if available
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        print("the fine-tuned translation model is not found")
    # Load the tokenizer from the fine-tuned translation model
    tokenizer = NllbTokenizer.from_pretrained(model_name)
    
    def fix_tokenizer(tokenizer, new_lang='tru_Latn'):
        """ Add a new language token to the tokenizer vocabulary (this should be done each time after its initialization) """
        old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
        tokenizer.lang_code_to_id[new_lang] = old_len-1
        tokenizer.id_to_lang_code[old_len-1] = new_lang
        # always move "mask" to the last position
        tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

        tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
        tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
        if new_lang not in tokenizer._additional_special_tokens:
            tokenizer._additional_special_tokens.append(new_lang)
        # clear the added token encoder; otherwise a new token may end up there by mistake
        tokenizer.added_tokens_encoder = {}
        tokenizer.added_tokens_decoder = {}

    if len(tokenizer) != tokenizer.vocab_size: #Check whether the values between len(tokenizer) and tokenizer.vocab_size are same after we added Truku language tag
        # This is only performed when we already expanded the tokenizer of the NLLB model
        print("fix the tokenizer configuration")
        fix_tokenizer(tokenizer)
        
    print("number of parameters:", model.num_parameters())
    def batch_tokenize_fn(examples):
        """
        Generate the input_ids and labels field for dataset dict of training data.
        """
        #set source (input) and target Languages
        src = (list(examples.keys()))[0] # get the language of source (input)
        tgt = (list(examples.keys()))[1] # get the language of target (output)
        sources = examples[src] # get the input samples
        targets = examples[tgt] # get the target samples
            
        src = 'tru_Latn' if src == 'truku' else 'zho_Hant' # set the language tag of of source (input)
        tgt = 'tru_Latn' if tgt == 'truku' else 'zho_Hant' # set the language tag of of target (output)
        
        # tokenizing the input sentences
        tokenizer.src_lang = src #The language to use as source language for translation
        model_inputs = tokenizer(sources, max_length=config.max_source_length, truncation=True)
    
        # tokenizing the target sentences
        # tokenized ids of the target are stored as the labels field
        tokenizer.src_lang = tgt #The language to use as target language for translation
        labels = tokenizer(targets, max_length=config.max_target_length, truncation=True)
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    # Tokenizing sentence pair of Truku and Chinese dataset   
    tc_dataset_dict_tokenized = dataset_dict_tc.map(
            batch_tokenize_fn,
            batched=True,
            remove_columns=dataset_dict_tc["test"].column_names
        )
    ct_dataset_dict_tokenized = dataset_dict_ct.map(
            batch_tokenize_fn,
            batched=True,
            remove_columns=dataset_dict_ct["test"].column_names
        )
    print(ct_dataset_dict_tokenized["test"][0])
    # evalution metrics computation
    def compute_metrics(eval_pred):
        """
        Compute rouge, BERTscore, chrF, and bleu metrics for seq2seq model generated prediction.
        
        tip: we can run trainer.predict on our eval dataset to see what a sample
        eval_pred object would look like when implementing custom compute metrics function
        """
        predictions, labels = eval_pred
        # Decode prediction samples, which is in ids into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode tokenized labels a.k.a. reference translation into text
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
        chrf=chrf_score.compute(predictions=decoded_preds, references=decoded_labels) # The higher the value, the better the translations
        berts = bert_score.compute(predictions=decoded_preds, references=decoded_labels,  model_type="bert-base-chinese")
        result["bert_score"]= np.mean(berts['f1']) # The higher the value, the better the translations
        result["sacrebleu"] = score["score"] # The higher the value, the better the translations
        result["chrf"] = chrf["score"] # The higher the value, the better the translations
        return {k: round(v, 4) for k, v in result.items()}
    
    #The training arguments for the training session, can be ignored since we do not perform training session
    args = Seq2SeqTrainingArguments(output_dir="./")
    
    # Data collator used for seq2seq model
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        )
    
    # perform the testing process
    print('+++-----Testing stage for Truku to Chinese translation-----+++')
    pred_t2c=trainer.predict(tc_dataset_dict_tokenized["test"])
    print(pred_t2c[-1])
    print('+++-----Testing stage for Chinese to Truku translation-----+++')
    pred_c2t=trainer.predict(ct_dataset_dict_tokenized["test"])
    print(pred_c2t[-1])
if __name__ == "__main__":
    main()
