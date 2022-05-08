from app_logger import logging
from app_exception import AppException
from app_entity import DataValidationConfig, DataIngestionArtifact, DataValidationArtifact
import os
import sys


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20} ")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
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

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            status = self.is_train_test_file_exists()
            message = "Data validation status: {}".format(status)
            data_validation_artifact = DataValidationArtifact(is_validated=status,
                                                              message=message,
                                                              schema_file_path=self.data_validation_config.schema_file_path,
                                                              )
            logging.info(f"Data validation status: {status}.")
            logging.info(
                f"Data validation artifact: {data_validation_artifact}.")
            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Validation log ended.{'='*20} ")
