# Create translation dataset of Chinese and Truku languages from raw data
This is a repository for creating bidirectional translation dataset between Truku and Chinese languages (Truku ↔ Chinese) with a preprocessing step.
The new  translation dataset will contain prefixes in the input sentence such as 將華語成太魯閣族語 (Chinese→Truku) or 將太魯閣族語成華語 (Truku→Chinese).
Please make sure that the following files are already placed in this directory:
* `其他來源`
* `字根句型辭典`
* `族語E樂園`
* `聖經新舊約`

Then, run the `preprocessing_data.py` script.
