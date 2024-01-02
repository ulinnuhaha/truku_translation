import gradio as gr
# import the Translator class & Direction variable from  translation_model.py
from translation_model import Translator, DIRECTION
DIRECTION_LIST = list(DIRECTION.keys()) # get the keys of Direction dict

# The function to wrap an interface around
def translate_wrapper(dir, text_input, by_sentence=True, preprocess=True, random=False, num_beams=4):
    direction = DIRECTION.get(dir) #the translation direction
    
    # Calling the translator's class method in translation_model.py 
    result = translator.translate(
        text=text_input, # input sentence for translation
        direction=direction, 
        do_sample=random, #parameter to enable decoding strategies in model.generate()
        num_beams=int(num_beams), #Number of beams for beam search in model.generate()
        by_sentence=by_sentence, #True or false
        preprocess=preprocess, # Whether performing the preprocessing stage for the input sentence or not
    )
    return result
    
# value for article parameter in Gradio interface
note = """
This is a NLLB-200-600M model fine-tuned for translation between Truku and traditional Chinese languages.
"""

# set the Gradio interface
interface = gr.Interface(
    translate_wrapper, # call the wrapper function
    [
        gr.Dropdown(DIRECTION_LIST, type="value", label='Translation direction', value=DIRECTION_LIST), # Dropdown menu for translation direction
        gr.Textbox(label="Input Text", lines=2, placeholder='text to translate '), # Textbox for the input sentence
        gr.Checkbox(label="by sentence", value=True), # Checkbox whether to process the input text by splitting the sentence into some array or not 
    ],
    "text", # the data type for the outputs
    title='Truku-Chinese translation', # a title for the interface
    article=note, # An expanded article explaining the interface; if provided, appears below the input and output components
)


if __name__ == '__main__':
    translator = Translator()

    interface.launch()
