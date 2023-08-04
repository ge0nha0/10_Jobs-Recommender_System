import pandas as pd
import numpy as np
import re


class ppc_tool:

    def __init__(self):
        pass

        # Regex Filtering #
    def ppc_time(self,x):
        p = re.compile('[0-9].*년|[0-9].*(\.|월|\/|일)|[0-9].*(\:|시|분)')
        return p.sub('',x)

    def ppc_e_address(self,x):
        p = re.compile('[a-zA-Z0-9]*@[a-zA-Z0-9]*\.[a-zA-Z0-9]*|((https?://|www|WWW).*?( |$))')
        return p.sub('',x)

    def ppc_phone(self,x):
        p = re.compile('[0-9]{2,3}-[0-9]{3,4}-[0-9]{4}')
        return p.sub('',x)

    def ppc_tag(self,x):
        p = re.compile('&#39|&gt|&lt|&quot|&apos|#NAME?')
        return p.sub('',x)
    
    def ppc_special(self,x):
        p = re.compile('[^가-힣0-9a-zA-Z\s]|\r|\n|\t')
        return p.sub(' ',x)

class ppc_job_simple(ppc_tool):
    def __init__(self, df):
        self.df = df

    def ppc_job(self):
        df = self.df    
        df = self.ppc_nan(df)
        df['회사명'] = df['회사명'].apply(self.ppc_co_name)
        df = self.ppc_wage(df)
        df = self.ppc_domain(df)
        df.drop(columns=['직종코드','직종코드3','임금형태','최소임금액','최대임금액'],inplace=True)
        return df

    # 결측치 현황을 제공하고 선택적으로 결측치를 처리할 수 있는 함수 # 
    def ppc_nan(self, df):
        nan_col = df.columns[df.isna().any()].tolist()
        for col in nan_col:
            print({col:df[col].value_counts(dropna=False)})
            print('최빈값 대체 = 1 / 삭제 = 2')
            if input() == '1':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df.dropna(subset=[col], inplace=True)
        return df

    # 회사명 간소화 및 통일 # 
    def ppc_co_name(self, co_name):
        p = re.compile('\(주\)|주식회사|\(사\)|\(유\)|\(합\)|\(자\)|\(협\)|\(의\)')
        return p.sub('',co_name).strip()

    # 임금 단일화 및 전처리 # 
    def ppc_wage(self, df):
        df['최대임금액'] = df['최대임금액'].astype('int')
        df['최소임금액'] = df['최소임금액'].astype('int')
        df['평균임금'] = df.apply(self.avg_wage,axis=1)
        df['평균임금'] = df.apply(self.unify_wage,axis=1)
        df['평균임금'] = round(df['평균임금'],-4)
        df['평균임금'] = df['평균임금'].astype('string')  
        return df

    # 평균 임금 추출 #
    def avg_wage(self, df):
        if df['최대임금액'] != 0:
            return (df['최소임금액'] + df['최대임금액']) / 2
        else:
            return df['최소임금액']

    # 시급, 연봉, 일급 -> 월급으로 통일 #
    def unify_wage(self, df):
        if df['임금형태'] == '시급':
            if df['근무형태'] == '주6일근무':
                return df['평균임금'] * 6/7 * 30.5 * 8
            else:
                return df['평균임금'] * 5/7 * 30.5 * 8
        elif df['임금형태'] == '일급':
            if df['근무형태'] == '주6일근무':
                return df['평균임금'] * 6/7 * 30.5
            else:
                return df['평균임금'] * 5/7 * 30.5
        elif df['임금형태'] == '연봉':
            return df['평균임금'] / 12
        else:
            return df['평균임금']
        
    def ppc_domain(self,df):
        domain_code = pd.read_csv('data/raw_data/직종코드.csv',encoding='mbcs',dtype='string')[['직종코드3','직종명3','직종명2','직종명1']]
        df = df.merge(domain_code,how='inner',left_on='직종코드',right_on='직종코드3') #직종코드 사전에 없는 직종코드는 삭제
        df.dropna(subset=['직종코드3'],inplace=True)
        df['직종명3'] = df['직종명3'].apply(self.ppc_special)
        df['직종명2'] = df['직종명2'].apply(self.ppc_special)
        df['직종명1'] = df['직종명1'].apply(self.ppc_special)
        return df 
        

class ppc_job_specific(ppc_tool):

    def __init__(self,df):
        self.df = df

    def ppc_job_specific(self):
        df = self.df
        df = self.ppc_nan(df)

        df['모집직종'] = df['모집직종'].apply(self.ppc_area)
        df['구인제목'] = df['구인제목'].apply(self.ppc_title)
        df['관련직종'] = df['관련직종'].apply(self.ppc_related_area)
        df['직무내용'] = df['직무내용'].apply(self.ppc_content)
        df['학력'] = df['학력'].apply(self.ppc_education)
        df['외국어'] = df['외국어'].apply(self.ppc_foreign_language)
        df['자격면허'] = df['자격면허'].apply(self.ppc_license)
        df['우대조건'] = df['우대조건'].apply(self.ppc_preferential)
        
        df['기타안내'] = df['기타안내'].apply(self.ppc_info_etc)
        df['고용형태'] = df['고용형태'].apply(self.ppc_work_form)
        df['기타우대조건'] = df['기타우대조건'].apply(self.ppc_preferential_etc)
        df['인근전철역'] = df['인근전철역'].apply(self.ppc_subway)
        df['근무시간/형태'] = df['근무시간/형태'].apply(self.ppc_work__time)
        return df

    # 결측치 처리 #
    def ppc_nan(self, df):
        df.fillna('', inplace=True)
        return df
    
    # 모집직종 #
    def ppc_area(self,job_area):
        p = re.compile('\([0-9]*\)$') 
        job_area = p.sub('',job_area)
        job_area = self.ppc_special(job_area)
        return job_area.replace('  ',' ')

    # 구인제목 #
    def ppc_title(self,job_title):
        job_title = self.ppc_time(job_title)
        job_title = self.ppc_special(job_title)
        return job_title.replace('  ',' ')
    
    # 관련 직종 #
    def ppc_related_area(self, job_related_area):
        job_related_area = self.ppc_special(job_related_area)
        return job_related_area.replace('  ',' ')
    
    # 직무 내용 # 
    def ppc_content(self, job_content):
        job_content = self.ppc_time(job_content)
        job_content = self.ppc_tag(job_content)
        job_content = self.ppc_phone(job_content)
        job_content = self.ppc_e_address(job_content)
        job_content = self.ppc_special(job_content)
        return job_content.replace('  ',' ') #숫자를 싹다 밀어버릴지 고민

    # 고용형태 # -> alert
    def ppc_work_form(self, job_work_form):
        x = job_work_form.split('/')
        if x[0].startswith('기간의 정함이 있는'):
            x[0] = '정규직'
        elif x[0].startswith('기간의 정함이 없는'):
            x[0] = self.ppc_special(x[0]).replace(' ','')
            x[0] = x[0].replace('기간의정함이없는근로계약','비정규직 ')

        for i in range(1,len(x)):
            x[i] = x[i].replace(' ','')
        x = ' '.join(x)
        return self.ppc_special(x)

    # 학력 #
    def ppc_education(self, job_education):
        p = re.compile('\-.*')
        return p.sub('',job_education)

    # 외국어 # 
    def ppc_foreign_language(self, job_foreign_language):
        p = re.compile('\((상|중|하)\)|')
        return p.sub('',job_foreign_language).replace(',',' ')
    
    # 전공 #
    def ppc_major(self, job_major):
        p = re.compile('\(.*?\)')
        job_major = p.sub('',job_major)
        return self.ppc_special(job_major)

    # 자격면허 # -> alert
    def ppc_license(self, job_license):
        p = re.compile('\(.*?\)')
        job_license = p.sub('',job_license)
        return self.ppc_special(job_license)

    # 우대조건 #
    def ppc_preferential(self, job_preferential):
        p = re.compile('\((준|50세이상)\)')
        return p.sub('',job_preferential).replace(',',' ')

    # 기타우대조건 # -> alert
    def ppc_preferential_etc(self, job_preferential_etc):
        job_preferential_etc = self.ppc_e_address(job_preferential_etc)
        job_preferential_etc = self.ppc_phone(job_preferential_etc)
        job_preferential_etc = self.ppc_time(job_preferential_etc)
        job_preferential_etc = self.ppc_special(job_preferential_etc)
        # Stopword 자리 or 전체 데이터에 대해서 한 번에 진행해도됨
        return job_preferential_etc

    # 기타안내
    def ppc_info_etc(self,job_info_etc):
        job_info_etc = self.ppc_time(job_info_etc)
        job_info_etc = self.ppc_e_address(job_info_etc)
        job_info_etc = self.ppc_phone(job_info_etc)
        job_info_etc = self.ppc_tag(job_info_etc)
        return self.ppc_special(job_info_etc).replace('  ',' ')

    # 인근지하철 # -> alert
    def ppc_subway(self, job_subway):
        p = re.compile('[0-9]호선|[0-9].*출구|[0-9].*M')
        job_subway = p.sub('',job_subway)
        return self.ppc_special(job_subway).strip()

    # 근무시간
    def ppc_work__time(self, job_work_time):
        p = re.compile('주.*[0-9]일.*근무')
        job_work_time = p.sub('',job_work_time)
        p = re.compile('[0-9]{2}:[0-9]{2}')
        job_work_time = p.sub(lambda x: x.group().replace(':','시 ')+'분',job_work_time)
        job_work_time = self.ppc_tag(job_work_time)
        job_work_time = self.ppc_special(job_work_time)
        p = re.compile('평균근무시간')
        return p.sub('',job_work_time)
    
    ######### temp code #########
    # 직종코드 따오는 코드 만들기전 feature에서 추출해서 사용
    def ppc_domain(self, job_area):    
        p = re.compile('[0-9]{6}')
        job_area = p.findall(job_area)[-1]
        return job_area
            
    
    # Regex Filtering #
    def ppc_time(self,x):
        p = re.compile('[0-9].*년|[0-9].*(\.|월|\/|일)|[0-9].*(\:|시|분)')
        return p.sub('',x)

    def ppc_e_address(self,x):
        p = re.compile('[a-zA-Z0-9]*@[a-zA-Z0-9]*\.[a-zA-Z0-9]*|((https?://|www|WWW).*?( |$))')
        return p.sub('',x)

    def ppc_phone(self,x):
        p = re.compile('[0-9]{2,3}-[0-9]{3,4}-[0-9]{4}')
        return p.sub('',x)

    def ppc_tag(self,x):
        p = re.compile('&#39|&gt|&lt|&quot|&apos|#NAME?')
        return p.sub('',x)
    
    def ppc_special(self,x):
        p = re.compile('[^가-힣0-9a-zA-Z\s]|\r|\n|\t')
        return p.sub(' ',x)

class geo_data_generator:
    
    def __init__(self,df):
        self.df = df

    def geo_data_generator(self):
        # geo = pd.read_csv(self.path,encoding='mbcs')
        geo_site = self.df
        geo_site.fillna('',inplace=True)
        geo_dict = {'경상남도':'경남','경상북도':'경북','전라남도':'전남','전라북도':'전북','충청남도':'충남','충청북도':'충북'}
        for i in geo_dict.keys():
            geo_site['시도'] = geo_site['시도'].str.replace(i,geo_dict[i])
        geo_site['시도'] = geo_site['시도'].str[:2]
        geo_site['행정구역'] = (geo_site['시도'] + ' ' + geo_site['시군구'] + ' ' + geo_site['읍면동/구'] + ' ' + geo_site['읍/면/리/동'] + ' ' + geo_site['리']).str.strip()
        geo_site = geo_site[['행정구역','위도','경도']]
        return geo_site
