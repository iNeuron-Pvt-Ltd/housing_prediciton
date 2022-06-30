from app_logger import logging
from app_exception import AppException
from app_entity import DataValidationConfig, DataIngestionArtifact, DataValidationArtifact
import pandas as pd

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
import os
import sys
import json


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'=' * 20}Data Validation log started.{'=' * 20} ")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            dir_name = os.path.dirname(self.data_validation_config.report_page_file_path)
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            raise AppException(e, sys) from e

    def is_train_test_file_exists(self) -> bool:
        try:
            is_train_file_exist = False
            is_test_file_exist = False
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            logging.info(f"Checking if train file exists: {train_file_path}.")
            if os.path.exists(train_file_path):
                is_train_file_exist = True
                logging.info(f"Train file exists: {train_file_path}.")
            else:
                logging.info(f"Train file does not exists: {train_file_path}.")

            logging.info(f"Checking if test file exists: {test_file_path}.")
            if os.path.exists(test_file_path):
                is_test_file_exist = True
                logging.info(f"Test file exists: {test_file_path}.")
            else:
                logging.info(f"Test file does not exists: {test_file_path}.")
            return is_train_file_exist and is_test_file_exist
        except Exception as e:
            raise AppException(e, sys) from e

    def get_train_test_data_frame(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df,test_df
        except Exception as e:
            raise  AppException(e,sys) from e

    def save_report_page_dashboard(self):
        try:
            train_df, test_df = self.get_train_test_data_frame()
            dashboard = Dashboard(tabs=[DataDriftTab()])
            dashboard.calculate(train_df,test_df)
            dashboard.save(self.data_validation_config.report_page_file_path)
        except Exception as e:
            raise  AppException(e,sys) from e

    def save_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df, test_df = self.get_train_test_data_frame()

            profile.calculate(train_df, test_df)

            report = json.loads(profile.json())

            with open(self.data_validation_config.report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
        except Exception as e:
            raise  AppException(e,sys) from e

    def validate_data_drift(self):
        try:
            self.save_report_page_dashboard()
            self.save_report()
        except Exception as e:
            raise AppException(e, sys)from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            status = self.is_train_test_file_exists()
            self.validate_data_drift()
            message = "Data validation status: {}".format(status)
            data_validation_artifact = DataValidationArtifact(is_validated=status,
                                                              message=message,
                                                              schema_file_path=self.data_validation_config.schema_file_path,
                                                              report_page=self.data_validation_config.report_page_file_path,
                                                              report=self.data_validation_config.report_file_path,
                                                              validator_obj=self,
                                                              )
            logging.info(f"Data validation status: {status}.")
            logging.info(
                f"Data validation artifact: {data_validation_artifact}.")
            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Data Validation log ended.{'=' * 20} ")
