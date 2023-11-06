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
python mt5-standard.py \
  --model_checkpoint google/mt5-small \
  --train_file ./datasets/train_chi2tru.tsv \
  --eval_file ./datasets/val_chi2tru.tsv \
  --cache_dir ./1d_translation_model \
  --trans_direction chi2tru
```
