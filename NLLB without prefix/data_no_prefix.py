import pandas as pd
from sklearn.model_selection import train_test_split

#Load the raw dataset
def get_data(file,direct):
    df=pd.read_excel('./dataset/'+file+'.xlsx') # read file
    df =df.replace(to_replace=r'_x000D_', value='', regex=True) #delete _x000D_ found in the element
    df=df.rename(columns={"華語": "chinese", "太魯閣族語": "truku"}) #rename columns
    df=df.dropna() #drop Nan Values
    if direct== 'chi2tr': #create dataset where Chinese as source language 
        return df[['chinese','truku']]
    else: # #create dataset where Truku as source language 
        return df[['truku','chinese']]
out_dir="./dataset/" # Output directory
# Create bilingual dataset for translation
def create_bitext_data(direct):
    # read file names
    t1=get_data('其他來源',direct)
    t2=get_data('字根句型辭典',direct)
    t3=get_data('族語E樂園',direct)
    t4=get_data('聖經新舊約',direct) #bible 
    
    data_all = pd.concat([t1,t2,t3,t4]) #group all dataset
    data_all=data_all.drop_duplicates(subset=['truku']) #drop duplicate row

    train, test = train_test_split(data_all,test_size=0.2,train_size=0.8,shuffle=True) #split for training and testing data
    train, val = train_test_split(train,test_size=0.2,train_size=0.8,shuffle=True) #split for training and validation data
    return train, test, val
    
# Create dataset: Chinese to Truku
print('Starting to create: Chinese to Truku dataset')
direct='chi2tr' # the translation direction of Chinese to Truku
ct_train, ct_test, ct_val = create_bitext_data(direct)
#Save the data to tsv files
ct_train.to_csv(out_dir+'ct_train.tsv', sep="\t", index=False)
ct_val.to_csv(out_dir+'ct_val.tsv', sep="\t", index=False)
ct_test.to_csv(out_dir+'ct_test.tsv', sep="\t", index=False) #testing data for Truku to Chinese translation

# Create dataset: Truku to Chinese
print('Starting to create: Truku to Chinese dataset')
direct='tr2chi'  # the translation direction of Truku to Chinese
tc_train, tc_test, tc_val = create_bitext_data(direct)
#Save the data to tsv files
tc_train.to_csv(out_dir+'tc_train.tsv', sep="\t", index=False)
tc_val.to_csv(out_dir+'tc_val.tsv', sep="\t", index=False)
tc_test.to_csv(out_dir+'tc_test.tsv', sep="\t", index=False) #testing data for Chinese to Truku translation
print('Successfully saving all dataset')
