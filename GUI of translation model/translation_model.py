# import the libraries
import re
import sys
import typing as tp
import unicodedata
import torch
from sacremoses import MosesPunctNormalizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

#the address of the fine-tuned translation model
MODEL_URL = "./pretrained_model/nllb_expanded_tr_ch"

# The prefix scheme in the input sentence for each translation direction
DIRECTION = {
    "Truku to Traditional Chinese": "將太魯閣族語成華語: ",
    "Traditional Chinese to Truku": "將華語成太魯閣族語: ",
}

# replace non-printable characters
def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

# A class for pre-processing input texts
class TextPreprocessor:
    """
    Mimic the text preprocessing made for the NLLB model.
    This code is adapted from the Stopes repo of the NLLB team:
    https://github.com/facebookresearch/stopes/blob/main/stopes/pipelines/monolingual/monolingual_line_processor.py#L214
    """

    def __init__(self, lang="en"):
        self.mpn = MosesPunctNormalizer(lang=lang)
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = get_non_printing_char_replacer(" ")

    def __call__(self, text: str) -> str:
        clean = self.mpn.normalize(text) #normalize the punctuation
        clean = self.replace_nonprint(clean) # replace non printable characters
        # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
        clean = unicodedata.normalize("NFKC", clean) # Return the normal form for the Unicode string. NFKC = Normalization Form Compatibility Composition
        return clean

# Split each sentence in the input texts in conjunction with getting the dividing mark
def sentenize_with_fillers(text, fix_double_space=True, ignore_errors=False):
    """Apply a sentence splitter and return the sentences and all separators before and after them"""
    if fix_double_space:
        text = re.sub(" +", " ", text) # remove double space
    sentences = re.findall(r"[^.!?]+", text) # split the sentence sentence with the punctuation as the splitter
    fillers = []
    i = 0
    # Get the splitter or filler among the sentences
    for sentence in sentences:
        start_idx = text.find(sentence, i)
        if ignore_errors and start_idx == -1:
            # print(f"sent not found after {i}: `{sentence}`")
            start_idx = i + 1
        assert start_idx != -1, f"sent not found after {i}: `{sentence}`"
        fillers.append(text[i:start_idx])
        i = start_idx + len(sentence)
    fillers.append(text[i:])
    return sentences, fillers

# A class for the main translation process
class Translator:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_URL) # get the translation model
        if .cuda.is_available():
            self.model.cuda()
        self.tokenizer = NllbTokenizer.from_pretrained(MODEL_URL) # get the tokenizer
        self.preprocessor = TextPreprocessor() 

    # the method for translation
    def translate(
        self,
        text, # input sentence
        direction, # the translation direction
        max_length="auto", # the maximum length of the tokenized input
        num_beams=4, #Number of beams for beam search in model.generate()
        by_sentence=True, # splitting the sentences into some arrays
        preprocess=True, #  perform preprocessing for the input sentence or not
        **kwargs, # others arguments in the generating output text
    ):
        """Translate a text sentence by sentence, preserving the fillers around the sentences."""
        if by_sentence: # if True for splitting the sentences into some arrays
            sents, fillers = sentenize_with_fillers(
                text, ignore_errors=True
            )
        else:
            sents = [text]
            fillers = ["", ""]
        #process “unknown symbol” and non-standard punctuation marks
        if preprocess: # if True for performing preprocessing for the input sentence
            sents = [self.preprocessor(sent) for sent in sents]
        results = []
        for sent, sep in zip(sents, fillers):
            results.append(sep)
            results.append(
                self.translate_single(
                    direction+sent, # given a prefix based on the translation direction for the input sentence
                    max_length=max_length,
                    num_beams=num_beams,
                    **kwargs,
                )
            )
        results.append(fillers[-1])
        return "".join(results)

    def translate_single(
        self,
        text,
        max_length="auto",
        num_beams=4,
        n_out=None,
        **kwargs,
    ):
        # Tokenzing the input sentences
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        if max_length == "auto":
            max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
            
        # Generating the output text of the translation
        generated_tokens = self.model.generate(
            **encoded.to(self.model.device),
            #forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=n_out or 1,
            **kwargs,
        )
        out = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        if isinstance(text, str) and n_out is None:
            return out[0]
        return out


if __name__ == "__main__":
    print("Initializing a translator to pre-download models...")
    translator = Translator()
    print("Initialization successful!")
