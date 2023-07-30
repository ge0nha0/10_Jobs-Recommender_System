import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
from path import path_controller

class JobParser(path_controller):
    def __init__(self, args):
        super().__init__()
        self.end_point = args.end_point
        self.service_key = f'authKey={args.service_key}'
        self.call_tp = None
        self.return_type = f'&returnType={args.return_type}'
        self.job_simple_path = self._get_rawdata_datasets_path()[0]
        self.job_specific_path = self._get_rawdata_datasets_path()[1]

    def save_file(self, df, path):
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)
        if path.is_file():
            print('Already exists. Skip saving')
            return
        else:
            df.to_csv(path, index=False, encoding='utf-8-sig')
        return

class job_simple(JobParser):
    def __init__(self, args):
        super().__init__(args)
        self.row = []

    def parse_want_L(self, wanted,):
        try:
            wantedauthno = wanted.find("wantedAuthNo").get_text()
            company = wanted.find("company").get_text()
            sal_tpnm = wanted.find("salTpNm").get_text()
            minsal = wanted.find("minSal").get_text()
            maxsal = wanted.find("maxSal").get_text()
            region = wanted.find("region").get_text()
            holiday_tpnm = wanted.find("holidayTpNm").get_text()
            career = wanted.find("career").get_text()
            jobscd = wanted.find("jobsCd").get_text()

            return {
                "구인인증번호" : wantedauthno,
                "회사명" : company,
                "임금형태" : sal_tpnm,
                "최소임금액" : minsal,
                "최대임금액" : maxsal,
                "근무지역" : region,
                "근무형태" : holiday_tpnm,
                "경력" : career,
                "직종코드" : jobscd
            }
        except AttributeError as e:
            return {
                "구인인증번호" : None,
                "회사명" : None,
                "임금형태" : None,
                "최소임금액" : None,
                "최대임금액" : None,
                "근무지역" : None,
                "근무형태" : None,
                "경력" : None,
                "직종코드" : None
            }

    def parse_jobs_L(self):
        self.call_tp = "&callTp=L"
        display = "&display=10"
        print('self.end_point + self.service_key + self.call_tp + self.return_type + start_page + display')
        for page in range(1, 2):
            start_page = f"&startPage={page}"
            result = requests.get(
                self.end_point + self.service_key + self.call_tp + self.return_type + start_page + display
            )
            
            soup = BeautifulSoup(result.text, 'lxml-xml')
            wanteds = soup.find_all("wanted")

            for wanted in wanteds:
                self.row.append(self.parse_want_L(wanted))

        df = pd.DataFrame(self.row)
        self.save_file(df, self.job_simple_path)
        return df
        
        
        
class job_specific(JobParser):

    def __init__(self, args, authno_list):
        super().__init__(args)
        self.row = []
        self.authno_list = authno_list

    def parse_want_D(self, wanted, Authno):
        try:
            jobsnm = wanted.find('jobsNm').get_text()
            wantedtitle = wanted.find('wantedTitle').get_text()
            reljbosnm = wanted.find('relJobsNm').get_text()
            jobcont = wanted.find('jobCont').get_text()
            emptpnm = wanted.find('empTpNm').get_text()
            edunm = wanted.find('eduNm').get_text()
            forlang = wanted.find('forLang').get_text()
            major = wanted.find('major').get_text()
            certificate = wanted.find('certificate').get_text()
            compabl = wanted.find('compAbl').get_text()
            pfcond = wanted.find('pfCond').get_text()
            etcpfcond = wanted.find('etcPfCond').get_text()
            etchopecont = wanted.find('etcHopeCont').get_text()
            nearline = wanted.find('nearLine').get_text()
            workday = wanted.find('workdayWorkhrCont').get_text()

            return {
                "구인인증번호" : Authno,
                "모집직종" : jobsnm,
                "구인제목" : wantedtitle,
                "관련직종" : reljbosnm,
                "직무내용" : jobcont,
                "고용형태" : emptpnm,
                "학력" : edunm,
                "외국어" : forlang,
                "전공" : major,
                "자격면허" : certificate,
                "컴활능력" : compabl,
                "우대조건" : pfcond,
                "기타우대조건" : etcpfcond,
                "기타안내" : etchopecont,
                "인근전철역" : nearline,
                "근무시간/형태" : workday
            }
        except AttributeError as e:
            return {
                "모집직종" : None,
                "구인제목" : None,
                "관련직종" : None,
                "직무내용" : None,
                "고용형태" : None,
                "학력" : None,
                "외국어" : None,
                "전공" : None,
                "자격면허" : None,
                "컴활능력" : None,
                "우대조건" : None,
                "기타우대조건" : None,
                "기타안내" : None,
                "인근전철역" : None,
                "근무시간/형태" : None
            }

    def parse_jobs_D(self):
        self.call_tp = "&callTp=D"
        infosvc = "&infoSvc=VALIDATION"
        for Authno in self.authno_list:
            wantedAuthno = f"&wantedAuthNo={Authno}"
            result = requests.get(self.end_point + self.service_key + self.call_tp + self.return_type + wantedAuthno + infosvc)
            soup = BeautifulSoup(result.text, 'lxml-xml')
            wanteds = soup.find_all("wantedInfo")

            for wanted in wanteds:
                self.row.append(self.parse_want_D(wanted, Authno))

        df = pd.DataFrame(self.row)
        self.save_file(df, self.job_specific_path)
        return df 
       