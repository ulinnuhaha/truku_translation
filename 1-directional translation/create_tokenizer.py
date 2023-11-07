import pandas as pd
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset
# load the corpus we have
def get_data(file):
    df=pd.read_excel('./datasets/'+file+'.xlsx')
    df =df.replace(to_replace=r'_x000D_', value='', regex=True) #delete _x000D_ found in the element
    df=df.rename(columns={"華語": "chinese", "太魯閣族語": "truku"}) #rename columns
    return df[['chinese','truku']]

df1=get_data('其他來源')
df2=get_data('字根句型辭典')
df3=get_data('族語E樂園')
df4=get_data('聖經新舊約') #bible

df=pd.concat([df1,df2,df3,df4])
df=df.drop_duplicates() #drop duplicates of rows
#Get The Truku dataset & Chinese from our dataset
truku_data=df['truku'].tolist()
chin_data=df['chinese'].tolist()
tr_ch=truku_data + chin_data

#load sentence-piece-unigram-based tokenizer
from mt5_tokenizer_model import SentencePieceUnigramTokenizer
vocab_size = 16_000
input_sentence_size = None
tokenizer = SentencePieceUnigramTokenizer(unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

# Initialize dataset to create a new tokenizer
tc_data=pd.DataFrame(tr_ch,columns =['text'])
tc_data['text']=tc_data['text'].apply(lambda x: str(x))
tc_data=tc_data.dropna() #delete NaN Row
from datasets import Dataset
dataset1=Dataset.from_pandas(tc_data)
#dataset1=dataset1.remove_columns(["__index_level_0__"])

# Load other traditional Chinese corpus to be added into our corpus
dataset2 = load_dataset("jed351/Traditional-Chinese-Common-Crawl-Filtered", data_files="C4_Traditional_Chinese-00001-of-00008.jsonl", split="train")
dataset2 = dataset2.remove_columns(["url","timestamp", "content_language", "content_type"])

# Concatenate two dataset for TOkenization
final_data=concatenate_datasets([dataset1, dataset2])

# Build an iterator over this dataset
def batch_iterator(input_sentence_size=None):
    if input_sentence_size is None:
        input_sentence_size = len(dataset1)
    batch_length = 100
    for i in range(0, input_sentence_size, batch_length):
        yield final_data[i: i + batch_length]['text']

# Train tokenizer
tokenizer.train_from_iterator(
    iterator=batch_iterator(input_sentence_size=input_sentence_size),
    vocab_size=vocab_size,
    show_progress=True,
)
# Save tokenizer
tokenizer.save("./mt5-tr_ch/tokenizer.json")
from transformers import T5Config, MT5Config
# Create the model's configuration file
config = MT5Config.from_pretrained("google/mt5-small", vocab_size=tokenizer.get_vocab_size())
config.save_pretrained("./mt5-tr_ch")
