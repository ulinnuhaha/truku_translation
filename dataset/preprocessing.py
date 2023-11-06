import os
import pandas as pd
from sklearn.model_selection import train_test_split

#Load the raw dataset
def get_data(file,direct):
    df=pd.read_excel('./'+file+'.xlsx')
    df =df.replace(to_replace=r'_x000D_', value='', regex=True) #delete _x000D_ found in the element/
    df=df.rename(columns={"華語": "chinese", "太魯閣族語": "truku"}) #rename columns
    if direct== 'chi2tr': #create dataset where chinese as source language 
      df=df.rename(columns={"chinese": "source_lang", "truku": "target_lang"}) #rename columns
    else: # #create dataset where truku as source language 
      df=df.rename(columns={"truku": "source_lang", "chinese": "target_lang"}) #rename columns
    df=df.dropna() #drop Nan Values
    return df[['source_lang','target_lang']]

# Create bilingual dataset for translation
def create_bitext_data(direct):
  
    t1=get_data('其他來源',direct)
    t2=get_data('字根句型辭典',direct)
    t3=get_data('族語E樂園',direct)
    t4=get_data('聖經新舊約',direct) #bible
    
    if direct == 'chi2tr':
        prefix = '將華語成太魯閣族語: '
    else:
        prefix = '將太魯閣族語成華語: '
        
    data_all = train=pd.concat([t1,t2,t3,t4]) #group all dataset
    data_all['source_lang'] = prefix + data_all['source_lang'].astype(str) #change all element values to string
    data_all=data_all.drop_duplicates(subset=['target_lang']) #drop duplicate row

    train, test = train_test_split(data_all,test_size=0.2,train_size=0.8,shuffle=True) #split for training and testing data
    train, val = train_test_split(train,test_size=0.2,train_size=0.8,shuffle=True) #split for training and validation data
    return train, test, val
# Create dataset: Chinese to Truku
print('Starting to create: Chinese to Truku dataset')
direct='chi2tr'
ct_train, ct_val, ct_test = create_bitext_data(direct)
print('Successfully creating: Chinese to Truku dataset')

# Create dataset: Truku to Chinese
print('Starting to create: Truku to Chinese dataset')
direct='tr2chi'
tc_train, tc_val, tc_test = create_bitext_data(direct)
print('Successfully creating: Truku to Chinese dataset')
## Group all data
train = pd.concat([tc_train, ct_train])
train = train.sample(frac=1) #shuffle the data
val = pd.concat([tc_val, ct_val])
val = val.sample(frac=1) #shuffle the data

#Save the data to tsv
train.to_csv('train.tsv', sep="\t", index=False,header=False)
val.to_csv('val.tsv', sep="\t", index=False,header=False)
#We set the testing data in two files since the evaluation metrics for Chinese to Truku & Truku to Chinese is distinct
tc_test.to_csv('test_tru2chi.tsv', sep="\t", index=False,header=False) #testing data for Chinese to Truku translation
ct_test.to_csv('test_chi2tru.tsv', sep="\t", index=False,header=False) #testing data for Truku to Chinese translation
print('Successfully saving all dataset')
