from pathlib import Path
from datetime import datetime
from config import RAW_DATASET_ROOT_FOLDER, MODEL_RESULT_ROOT_FOLDER

class path_controller:

    def __init__(self):
        self.today = datetime.today().strftime('%Y%m%d')
    
    def _get_data_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_root_path(self):
        root = self._get_data_root_path()
        return root.joinpath('raw_data')
    
    def _get_geosite_datasets_path(self):
        folder = self._get_rawdata_root_path()
        return folder.joinpath('geo_site.csv')

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        folder_name = 'date_{}'.format(self.today)
        return root.joinpath(folder_name)
    
    def _get_rawdata_datasets_path(self):
        folder = self._get_rawdata_folder_path()
        job_simple_path = folder.joinpath('채용목록.csv')
        job_specific_path = folder.joinpath('채용상세.csv')
        return job_simple_path , job_specific_path
    
    def _get_preprocessed_root_path(self):
        root = self._get_data_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = 'date_{}'.format(self.today) #중요변수 들어가야함 
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        dataset = folder.joinpath('dataset.pkl')
        # cascade_dataset = folder.joinpath('cascade_dataset.pkl')
        return dataset
    

    # save model path #
    def _get_model_root_path(self):
        return Path(MODEL_RESULT_ROOT_FOLDER)
    
    def _get_model_folder_path(self):
        root = self._get_model_root_path()
        folder_name = 'date_{}'.format(self.today)
        return root.joinpath(folder_name)
    
    def _get_model_result_path(self):
        folder = self._get_model_folder_path()
        model = folder.joinpath('model.확장자')
        return model