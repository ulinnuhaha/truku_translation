# Bilingual translation of Chinese and Truku languages
This is a repository for a translation model between Truku and Chinese languages using LLMs. We build some pre-trained models from LLMs to perform bilingual translation between Truku and Chinese languages.

# Run the translation model
To perform the translation model with a specific pre-trained model, you can run the `run_trans.py` script as the following command:
```bash
python run_trans.py --model_name_or_path ./pretrained_model/nllb_tr_ch
```
* before running the script, please make sure that the model in `pretrained_model` path is already downloaded and extracted 
