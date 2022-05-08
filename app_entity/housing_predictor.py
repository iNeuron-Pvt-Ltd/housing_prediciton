



from app_exception import AppException
import os,sys
from app_util import load_object

class HousingPredictor:

    def __init__(self,model_dir:str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise AppException(e, sys) from e


    def get_latest_model_path(self):
        try:
            folder_name = list(map(int,os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir,f"{max(folder_name)}")
            file_name  =os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir,file_name)
            return latest_model_path
        except Exception as e:
            raise AppException(e, sys) from e

    def predict(self,X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            return model.predict(X)
        except Exception as e:
            raise AppException(e, sys) from e