## Run the pre-preprocessing stage to create pre-processed data
Before we perform the fine-tuning scheme of the NLLB model without prefixes in the input sentence, we need to prepare training, evaluation, and testing data. To implement these data, you can run the `data_no_prefix.py` script.

## Run the training model to fine-tune NLLB for bilingual translation
To perform the fine-tuning process of the NLLB model to create the translation model between Truku and Traditional Chinese languages, you can run the `model_train.py` script as the following command:
```bashmodel_train
python model_train.py \
  --model_checkpoint pretrained_model/nllb_expanded \
  --cache_dir ./pretrained_model \
  --data_dir ./dataset
```
please make sure that this directory has the sub-directory containing the expanded NLLB model, we name it `pretrained_model` as the example.
## Run the testing stage of the fine-tuned translation model
To perform the testing process of the translation model, we carry out the bilingual translation of both Truku➝Chinese and Chinese➝Truku with some evaluation metrics. You can run the `model_test.py` script by loading the fine-tuned models in the `pretrained_model` directory as the following command:
```bash
python model_test.py \
  --model_name_or_path ./pretrained_model/pretrained_model_tr_ch_no_prefix \
  --data_dir ./dataset
```
