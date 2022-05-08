from app_entity.artifact_entity import DataIngestionArtifact
from app_logger import logging
from app_exception import AppException
from app_entity import DataIngestionConfig,TrainingPipelineConfig
import sys,os
import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class DataIngestion:
    

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        DataIngestion Intialization
        data_ingestion_config: DataIngestionConfig 
        """
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise AppException(e, sys) from e



    def extract_tgz_file(self,tgz_file_path: str):
        """
        tgz_file_path: str
        Extracts the tgz file into the raw data directory
        Function returns None
        """
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            housing_tgz = tarfile.open(tgz_file_path)
            os.makedirs(raw_data_dir, exist_ok=True)
            logging.info(f"Extracting tgz file: {tgz_file_path} into dir: {raw_data_dir}")
            housing_tgz.extractall(path=raw_data_dir)
            housing_tgz.close()
        except Exception as e:
            raise AppException(e,sys) from e

    def download_housing_data(self):
        """
        Fetch housing data from the url
        
        """
        try:
            
            housing_file_url = self.data_ingestion_config.dataset_download_url
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            os.makedirs(tgz_download_dir, exist_ok=True)
            housing_file_name = os.path.basename(housing_file_url)
            tgz_file_path = os.path.join(tgz_download_dir, housing_file_name)
            logging.info(f"Downloading housing data from {housing_file_url} into file {tgz_file_path}")
            urllib.request.urlretrieve(housing_file_url,tgz_file_path)
            logging.info(f"Downloaded housing data from {housing_file_url} into file {tgz_file_path}")
            return tgz_file_path
        except Exception as e:
            raise AppException(e, sys) from e

    def split_data_as_train_test(self)->DataIngestionArtifact:
        try:
            data_ingestion_config = self.data_ingestion_config
            raw_data_dir = data_ingestion_config.raw_data_dir

            #there is only one file in tgz hence we can use os.listdir to get the file name
            file_name = os.listdir(raw_data_dir)[0]
            housing_file_path = os.path.join(raw_data_dir, file_name)
            
            
            #Reading csv file using pandas
            logging.info(f"Reading csv file: [{housing_file_path}]")
            housing_data_frame = pd.read_csv(housing_file_path)


            logging.info(f"Splitting data into train and test")

            #creating adding column as income category based on median income
            housing_data_frame["income_cat"] = np.ceil(housing_data_frame["median_income"] / 1.5)
            #updating the income category to 5.0 if it is less than 5
            housing_data_frame["income_cat"].where(housing_data_frame["income_cat"] < 5, 5.0, inplace=True)


            #Splitting data into train and test
            strat_train_set = None
            strat_test_set = None
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            for train_index, test_index in split.split(housing_data_frame, housing_data_frame["income_cat"]):
                strat_train_set = housing_data_frame.loc[train_index].drop(["income_cat"],axis=1)
                strat_test_set = housing_data_frame.loc[test_index].drop(["income_cat"],axis=1)

            #saving the train and test dataframes   
            train_file_path = os.path.join(data_ingestion_config.ingested_train_dir, file_name)
            test_file_path = os.path.join(data_ingestion_config.ingested_test_dir, file_name)

            if strat_train_set is not None:
                os.makedirs(data_ingestion_config.ingested_train_dir, exist_ok=True)
                
                logging.info(f"Exporting training datset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)
            
            if strat_test_set is not None:
                os.makedirs(data_ingestion_config.ingested_test_dir, exist_ok=True)
                
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
            test_file_path=test_file_path,
            is_ingested=True,
            message="Data Ingestion completed and data set has been splited into train and test")
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")

            return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        try:
            downloaded_file_path = self.download_housing_data()
            self.extract_tgz_file(tgz_file_path=downloaded_file_path)
            return self.split_data_as_train_test()
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")


        


