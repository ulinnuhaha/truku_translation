# Create data to perform MLM
import pandas as pd

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
df

df['truku']=df['truku'].apply(lambda x: str(x)) #make all data in string type
df['truku']=df['truku'].apply(lambda row: row.encode('ascii',errors='ignore').decode()) # remove chinese characters

def word_count(sentence):
  sentence = str(sentence).split()
  return len(sentence)
df['word_count'] = df['truku'].apply(word_count)
datafinal=df.loc[df['word_count'] > 3].reset_index() #we only select a sentence that contains more than 3 words

# split to train and validation data
train=datafinal[round(0.15*(len(datafinal))):]
val=datafinal[:round(0.15*(len(datafinal)))]

# Crating special data for MLM fine-tuning
def create_data_mlm(df,type):
    df = df.dropna() # drop Nan values
    df=df[['truku']] #select only Truku data
    df=df.rename(columns={"truku": "text"})
    df.reset_index(drop=True, inplace=True)
    df.to_csv("./datasets/"+type+'_truku.csv', index=False)

create_data_mlm(train,'train')
create_data_mlm(val,'val')
