from path import path_controller
from datetime import datetime
from .preprocessing import ppc_job_simple, ppc_job_specific, geo_data_generator
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import re
    
class make_dataset(path_controller):

    def __init__(self):
        self.today = datetime.today().strftime('%Y%m%d')

    def load_dataset(self):
        self.preprocessor()
        dataset_path  = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocessor(self):
        dataset_path = self._get_preprocessed_dataset_path()
        print(dataset_path)
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        
        elif not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        job_simple, job_specific, geo_site = self.import_file()
        job_simple = ppc_job_simple(job_simple).ppc_job()
        job_specific = ppc_job_specific(job_specific).ppc_job_specific()
        df = job_simple.merge(job_specific,how='inner',on='구인인증번호')
        df, label, label_to_index = self.make_label(df)
        df = self.make_setence(df)
        df = self.stopword(df)
        df = self.ppc_large_space(df)
        dataset = self.make_samples(df,label)
        train, val, test = self.split_dataset(dataset)
        dataset  = {'train':train,
                    'val':val,
                    'test':test,
                    'label_to_index':label_to_index}

        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        with dataset_path.open('wb') as f:
                pickle.dump(dataset, f)
        return 
    
    def split_dataset(self,dataset):
        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.2, random_state=42)
        return train, val, test
    
    def stopword(self,df):
        stopwords = self.import_stopword()
        df = df.apply(lambda x: self.make_stopword(stopwords,x))
        return df

    def import_stopword(self):
        with open('data/stopwords.txt','r',encoding='utf-8') as f:
            stopwords = f.read().replace('\n','')
            stopwords = stopwords.split(',')
        return stopwords
    
    def make_stopword(self,stopwords, x):
        tmp = []
        tok = x.split(' ')
        for word in tok:
            if word not in stopwords:
                tmp.append(word)
        x = ' '.join(tmp)
        return x

    def make_samples(self,df,label):
        dataset = []
        for i in zip(df,label):
            dataset.append(list(i))
        return dataset

    def make_setence(self,df):
        df.set_index('구인인증번호',inplace=True)
        df = df.apply(lambda x : ' '.join(x),axis=1)
        return df

    def make_label(self,df):
        label = df.pop('직종명1')
        label_to_idx = {u: i for i, u in enumerate(label.unique())} 
        idx_to_label = {i: u for i, u in enumerate(label.unique())} 
        label = label.map(label_to_idx)
        return df, label, label_to_idx

    def import_file(self):
        #path에 파일이 있으면 불러오기
        job_simple_path, job_specific_path = self._get_rawdata_datasets_path()
        geo_site_path = self._get_geosite_datasets_path()
        job_simple = pd.read_csv(job_simple_path,encoding='utf-8',dtype='string')
        job_specific = pd.read_csv(job_specific_path, encoding='utf-8')
        geo_site = pd.read_csv(geo_site_path,encoding='euc-kr')
        return job_simple, job_specific, geo_site

    def ppc_large_space(self,df):
        p = re.compile(' {2,9999999}')
        df = df.apply(lambda x: p.sub(' ',x))
        return df
