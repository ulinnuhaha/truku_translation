import pandas as pd
from sklearn.model_selection import train_test_split

#Load the raw dataset
def get_data(file,direct):
    df=pd.read_excel('./dataset/'+file+'.xlsx')
    df =df.replace(to_replace=r'_x000D_', value='', regex=True) #delete _x000D_ found in the element/
    df=df.rename(columns={"華語": "chinese", "太魯閣族語": "truku"}) #rename columns
    df=df.dropna() #drop Nan Values
    if direct== 'chi2tr': #create dataset where chinese as source language 
      #df=df.rename(columns={"chinese": "source_lang", "truku": "target_lang"}) #rename columns
        return df[['chinese','truku']]
    else: # #create dataset where truku as source language 
      #df=df.rename(columns={"truku": "source_lang", "chinese": "target_lang"}) #rename columns
        return df[['truku','chinese']]
out_dir="./dataset/"
# Create bilingual dataset for translation
def create_bitext_data(direct):
  
    t1=get_data('其他來源',direct)
    t2=get_data('字根句型辭典',direct)
    t3=get_data('族語E樂園',direct)
    t4=get_data('聖經新舊約',direct) #bible 
    
    data_all = train=pd.concat([t1,t2,t3,t4]) #group all dataset
    #data_all['source_lang'] = prefix + data_all['source_lang'].astype(str) # Put the prefix in each sentence of source language
    data_all=data_all.drop_duplicates(subset=['truku']) #drop duplicate row

    train, test = train_test_split(data_all,test_size=0.2,train_size=0.8,shuffle=True) #split for training and testing data
    train, val = train_test_split(train,test_size=0.2,train_size=0.8,shuffle=True) #split for training and validation data
    return train, test, val
    
# Create dataset: Chinese to Truku
print('Starting to create: Chinese to Truku dataset')
direct='chi2tr'
ct_train, ct_test, ct_val = create_bitext_data(direct)
#Save the data to tsv
ct_train.to_csv(out_dir+'ct_train.tsv', sep="\t", index=False)
ct_val.to_csv(out_dir+'ct_val.tsv', sep="\t", index=False)
ct_test.to_csv(out_dir+'ct_test.tsv', sep="\t", index=False) #testing data for Truku to Chinese translation

# Create dataset: Truku to Chinese
print('Starting to create: Truku to Chinese dataset')
direct='tr2chi'
tc_train, tc_test, tc_val = create_bitext_data(direct)
#Save the data to tsv
tc_train.to_csv(out_dir+'tc_train.tsv', sep="\t", index=False)
tc_val.to_csv(out_dir+'tc_val.tsv', sep="\t", index=False)
tc_test.to_csv(out_dir+'tc_test.tsv', sep="\t", index=False) #testing data for Chinese to Truku translation
print('Successfully saving all dataset')
