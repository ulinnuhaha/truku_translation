import gradio as gr


from translation import Translator, DIRECTION
DIRECTION_LIST = list(DIRECTION.keys())

def translate_wrapper(text, dir, by_sentence=True, preprocess=True, random=False, num_beams=4):
    direction = DIRECTION.get(dir)
    result = translator.translate(
        text=text,
        direction=direction,
        do_sample=random, #parameter to enables decoding strategies
        num_beams=int(num_beams),
        by_sentence=by_sentence,
        preprocess=preprocess,
    )
    return result


article = """
This is a NLLB-200-600M model fine-tuned for translation between Truku and traditional Chinese languages.

"""


interface = gr.Interface(
    translate_wrapper,
    [
        gr.Textbox(label="Input Text", lines=2, placeholder='text to translate '),
        gr.Dropdown(DIRECTION_LIST, type="value", label='Translation direction', value=DIRECTION_LIST),
        gr.Checkbox(label="by sentence", value=True),
    ],
    "text",
    title='Truku-Chinese translation',
    article=article,
)


if __name__ == '__main__':
    translator = Translator()

    interface.launch()
