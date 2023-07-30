from path import path_controller
from datetime import datetime
from .preprocessing import ppc_job_simple, ppc_job_specific, geo_data_generator
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
    
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
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        job_simple, job_specific, geo_site = self.import_file()
        job_simple = ppc_job_simple(job_simple).ppc_job()
        job_specific = ppc_job_specific(job_specific).ppc_job_specific()
        df = job_simple.merge(job_specific,how='inner',on='구인인증번호')
        df.drop(columns=['채용제목','임금형태','급여','최소임금액','최대임금액','등록일자','마감일자','고용형태코드','직종코드','모집인원','임금조건','병역특례채용희망','경력조건','전형방법','제출서류준비물','연금4대보험','퇴직금','장애인 편의시설','기타복리후생','근무예정지'],inplace=True)
        # geo_site = geo_data_generator(geo_site).geo_data_generator()
        # cascade_dataset = df.merge(geo_site,how='inner',left_on = '근무지역',right_on = '행정구역')
        df, label, label_to_index = self.make_label(df)
        df = self.make_setence(df)
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

    def make_samples(self,df,label):
        dataset = []
        for i in zip(df,label):
            dataset.append(list(i))
        return dataset

    def make_setence(self,df):
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
        job_simple = pd.read_csv(job_simple_path,encoding='utf-8')
        job_specific = pd.read_csv(job_specific_path, encoding='utf-8')
        geo_site = pd.read_csv(geo_site_path,encoding='euc-kr')
        return job_simple, job_specific, geo_site

