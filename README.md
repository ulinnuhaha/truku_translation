# Bilingual translation of Traditional Chinese and Truku languages
This is a repository for a bidirectional translation model between Truku and Traditional Chinese languages (Truku ↔ Chinese) using LLMs. We build some translation models from LLMs to perform bilingual translation between Truku and Traditional Chinese languages. In the experimental stage, we used three LLMs. They are:
* mT5-small (`google/mt5-small`)
* NLLB-200's distilled 600M (`facebook/nllb-200-distilled-600M`)
* mBART-50 (`facebook/mbart-large-50`)

Please first install the required packages by:
```
pip install -r requirements.txt
```
For the one-directional translation model (Truku➝ Chinese or Chinese➝Truku), please go to the `1-directional translation` directory.
## Upgrade of NLLB model by expanding the tokenizer vocab
If you want to upgrade the NLLB model by expanding the tokenizer vocabs from a language such as Traditional Chinese Language, You can run:
```
upgrade_nllb_tokenizer.ipynb
```
After saving the expanded model in the `pretrained_model` directory as the initial model_checkpoint for the fine-tuning process of the translation task. You can train this model for translation-task between Truku and Traditional Chinese languages using `model_train.py`.
This training process uses an adding prefix scheme in the input sentence as we perform in `/dataset/preprocessing.py`.
## Run the training model to fine-tune LLMs for bilingual translation
To perform the fine-tuning process of LLMs to create the translation model between Truku and Traditional Chinese languages, you can run the `model_train.py` script as the following command:
```bashmodel_train
python model_train.py \
  --model_checkpoint facebook/mbart-large-50 \
  --cache_dir ./pretrained_model \
  --data_dir ./dataset
```
## Run the testing stage of the fine-tuned translation model
To perform the testing process of the translation model, we carry out the bilingual translation of both Truku➝Chinese and Chinese➝Truku with some evaluation metrics. You can run the `model_test.py` script by loading the fine-tuned models in the `pretrained_model` directory as the following command:
```bash
python model_test.py \
  --model_name_or_path ./pretrained_model/nllb_tr_ch \
  --data_dir ./dataset
```
We take different evaluation metrics in the testing stage. For Truku➝Chinese translation, we exploit:
* BLEU
* BERTScore
* chrF

While, for Chinese➝Truku translation, we exploit:
* BLEU
* BERTScore
* Rouge-1
## Fine-tune the NLLB model for bilingual translation scheme without prefixes in the input sentence
All the models in this directory perform the translation scheme using prefixes such as `將華語成太魯閣族語: ` in the input sentence, including the NLLB model. 
Thus, if you want to fine-tune the NLLB model with a scheme that does not put prefixes in the input sentence, you can go to the directory of `NLLB without prefix`. This directory still also performs the bidirectional translation model between Truku and Traditional Chinese languages (Truku ↔ Chinese)

## Run the simple translation model inference
To perform the translation model with a specific pre-trained model, you can run the `run_trans.py` script as the following command:
```bash
python run_trans.py --model_name_or_path ./pretrained_model/nllb_tr_ch
```
* Before running the script, please make sure that the model in `pretrained_model` path is already available or please first running `model_train.py`
