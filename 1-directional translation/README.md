# One-directional translation of Chinese and Truku languages (Truku➝Chinese or Chinese➝Truku)
This is a repository for 1-directional translation model between Truku and Chinese languages (Truku➝Chinese or Chinese➝Truku) using mT5. We build some pre-trained models based on mT5 to perform translation between Truku and Chinese languages. In the experimental stage, we used three schemes. They are:
* mT5-standard (Google's mT5-small)
* mT5 from Scratch
* mT5+MLM

## Run the pre-preprocessing stage to create pre-processed data
To perform this preprocessing step, you can run the `pre-processing.py` script.

## Run mT5-standard (Google's mT5-small)
To perform the pre-training process of the translation model based on mT5, we fine-tune Google's mT5-small to our translation model. You can run the `mt5-standard.py` script as the following command:
```bash
python mt5_train.py \
  --model_checkpoint google/mt5-small \
  --train_file ./datasets/train_chi2tru.tsv \
  --eval_file ./datasets/val_chi2tru.tsv \
  --cache_dir ./1d_translation_model \
  --trans_direction ch2tr
```
## Run mT5+MLM
To perform adapting Multilingual Language Models to unseen languages with MLM-TUNING, we do MLM-fine-tuning of Google's mT5-small on Truku languages to get the initial model checkpoint for translation. You can run the `mt5_mlm.py` script as the following command:
```bash
python mt5_mlm.py.py \
python3 run_mt5_mlm_pytorch.py \
    --model_name_or_path="google/mt5-small" \
    --tokenizer_name="google/mt5-small" \
    --train_file="./dataset/train_truku.csv" \
    --validation_file="./dataset/val_truku.csv" \
    --max_seq_length="512" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir="1d_translation_model/mt5_mlm"
```
