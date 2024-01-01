# Run the GUI of the translation model
This is a repository for creating the GUI of the translation model between Truku and Chinese languages (Truku ↔ Chinese).
We use Gradio to build a simple inference of the translation model in GUI form. 
Please make sure that you already installed the required packages by:
```
pip install -r requirement.txt
```
Before running the translation model, this directory should have the sub-directory of the fine-tuned translation model as we named it with `pretrained_model`. For the fine-tuned model, we use `./pretrained_model/nllb_expanded_tr_ch`.
Then, you can run the `run_app.py` script as follows:

```
python run_app.py
```
