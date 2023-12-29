import re
import sys
import typing as tp
import unicodedata

import torch
from sacremoses import MosesPunctNormalizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

MODEL_URL = "nllb_expanded_tr_ch"

DIRECTION = {
    "Truku to Traditional Chinese": "將太魯閣族語成華語: ",
    "Traditional Chinese to Truku": "將華語成太魯閣族語: ",
}

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
        clean = self.mpn.normalize(text)
        clean = self.replace_nonprint(clean)
        # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
        clean = unicodedata.normalize("NFKC", clean)
        return clean


def sentenize_with_fillers(text, fix_double_space=True, ignore_errors=False):
    """Apply a sentence splitter and return the sentences and all separators before and after them"""
    if fix_double_space:
        text = re.sub(" +", " ", text)
    sentences = re.findall(r"[^.!?]+", text)
    fillers = []
    i = 0
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


class Translator:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_URL)
        if torch.cuda.is_available():
            self.model.cuda()
        self.tokenizer = NllbTokenizer.from_pretrained(MODEL_URL)
        self.preprocessor = TextPreprocessor()


    def translate(
        self,
        text,
        direction,
        max_length="auto",
        num_beams=4,
        by_sentence=True,
        preprocess=True,
        **kwargs,
    ):
        """Translate a text sentence by sentence, preserving the fillers around the sentences."""
        if by_sentence:
            sents, fillers = sentenize_with_fillers(
                text, ignore_errors=True
            )
        else:
            sents = [text]
            fillers = ["", ""]
        #process “unknown symbol” and non-standard punctuation marks
        if preprocess:
            sents = [self.preprocessor(sent) for sent in sents]
        results = []
        for sent, sep in zip(sents, fillers):
            results.append(sep)
            results.append(
                self.translate_single(
                    direction+sent,
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
        #self.tokenizer.src_lang = src_lang

        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        if max_length == "auto":
            max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
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
