# One-directional translation of Chinese and Truku languages (Truku➝Chinese or Chinese➝Truku)
This is a repository for 1-directional translation model between Truku and Chinese languages (Truku➝Chinese or Chinese➝Truku) using mT5. We build some translation models based on mT5 to perform translation between Truku and Chinese languages. In the experimental stage, we used three schemes. They are:
* mT5-standard (Google's mT5-small)
* mT5 from Scratch
* mT5+MLM

## Run the pre-preprocessing stage to create pre-processed data
Before we perform the fine-tuning process of the 1-directional translation model, we need to prepare training, evaluation, and testing data. To implement the preprocessing step, you can run the `pre-processing.py` script.

## Run mT5-standard (Google's mT5-small)
To perform the pre-training process of the translation model based on mT5, we fine-tune Google's mT5-small to our translation model. You can run the `mt5_train.py` script as the following command:
```bash
python mt5_train.py \
  --model_checkpoint google/mt5-small \
  --train_file ./datasets/train_chi2tru.tsv \
  --eval_file ./datasets/val_chi2tru.tsv \
  --cache_dir ./1d_translation_model \
  --trans_direction ch2tr
```
trans_direction refers to the direction of the translation model where ch2tr is Chinese➝Truku
## Run mT5+MLM
Before performing mT5+MLM and mT5 from Scratch, we need to prepare special data, i.e., training and validation data for these processes. Then you can run `create_data_for_mlm.py` script.

To perform adapting Multilingual Language Models to unseen languages with MLM-TUNING, we do MLM-fine-tuning of Google's mT5-small on Truku languages to get the initial model checkpoint for translation.
After we obtain training and validation data of Truku corpus for performing MLM-Tuning. You can run the `mt5_mlm.py` script as the following command:
```bash
python mt5_mlm.py \
    --model_name_or_path="google/mt5-small" \
    --tokenizer_name="google/mt5-small" \
    --train_file="./datasets/truku_train.csv" \
    --validation_file="./datasets/truku_val.csv" \
    --max_seq_length="512" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir="1d_translation_model/mt5_ft_mlm"
```
When the model of mT5+MLM is already trained. The next step we can build the translation model similar to mT5-standard by running `mt5_train.py` script as the following command:
```bash
python mt5_train.py \
  --model_checkpoint 1d_translation_model/mt5_ft_mlm \
  --train_file ./datasets/train_chi2tru.tsv \
  --eval_file ./datasets/val_chi2tru.tsv \
  --cache_dir ./1d_translation_model \
  --trans_direction ch2tr
```
## Run mT5 from Scratch
Since we propose the mT5 model from Scratch. We also need to create a new tokenizer from scratch. To perform this,  You can run the `create_tokenizer.py` script and save it to `mt5-tr_ch` directory.

After we prepare training and validation data of Chinese and Truku data from our corpus. You can run the `mt5_mlm.py` script to perform mT5 from Scratch as the following command:
```bash
python mt5_mlm.py \
    --config_name="google/mt5-small" \
    --tokenizer_name="mt5-tr_ch" \
    --train_file="./datasets/truku_chinese_train.csv" \
    --validation_file="./datasets/truku_chinese_val.csv" \
    --max_seq_length="512" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir="1d_translation_model/mt5_scratch"
```
When the model of mT5 from Scratch is already trained. The next step we can build the translation model similar to mT5-standard by running `mt5_train.py` script as the following command:
```bash
python mt5_train.py \
  --model_checkpoint 1d_translation_model/mt5_scratch \
  --train_file ./datasets/train_chi2tru.tsv \
  --eval_file ./datasets/val_chi2tru.tsv \
  --cache_dir ./1d_translation_model \
  --trans_direction ch2tr
```
