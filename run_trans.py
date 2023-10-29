import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', 
        type=str, required=True, help='Load pre-trained model for translation')
    args = parser.parse_args()
    return args

def main():

    config=get_args()
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def generate_translation(prefix,example):
        """print out the source, target and predicted raw text."""
        source = prefix + example
        input_ids = tokenizer(source)["input_ids"]
        input_ids = torch.LongTensor(input_ids).view(1, -1).to(model.device)
        generated_ids = model.generate(input_ids, max_new_tokens=20)
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print('Translation: ', prediction)
        print("")

    while True:
        os.system('clear')
        print("Enter 1 : Translate from Chinese to Truku")
        print("Enter 2 : Translate from Truku to Chinese")
        value=input("Type = ")
        print("")
        if value == str(1):
            text = input("Type your Chinese sentence: ")
            prefix = "將華語成太魯閣族語: "
            generate_translation(prefix, text)
            then = input("Enter: E to exit or any to continue = ")
            if then == "E":
                exit()
            else:
                continue
        elif value ==str(2):
            text = input("Type your Truku sentence: ")
            prefix = "將太魯閣族語成華語: "
            generate_translation(prefix, text)
            then = input("Enter: E to exit or any to continue = ")
            if then == "E":
                exit()
            else:
                continue
        else:
            continue

if __name__ == "__main__":
    main()
