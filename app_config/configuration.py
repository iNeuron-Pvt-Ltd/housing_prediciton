from app_logger import logging
from app_exception import AppException
import os
import sys
from app_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, \
    ModelEvaluationConfig
from app_entity import ModelPusherConfig, TrainingPipelineConfig
from app_util import read_yaml_file
from datetime import datetime
from pathlib import Path

ROOT_DIR = os.getcwd()

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
# Varibale declaration
# Data Ingestion related variables


DATASET_SCHEMA_COLUMNS_KEY = "columns"
DATASET_SCHEMA_TARGET_COLUMN_KEY = "target_column"
DATASET_SCHEMA_DOMAIN_VALUE_KEY = "domain_value"

DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY = "tgz_download_dir"
DATA_INGESTION_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"
# Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"

# Data Transformation related variables
DATA_TRANSFORMATION_ARTIFACT_DIR = "data_transformation"
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY = "add_bedroom_per_room"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_object_file_name"
# Model Training related variables

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"

# Model Evaluation related variables

MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_ARTIFACT_DIR = "model_evaluation"
# Model Pusher config key
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)


class AppConfiguration:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH,current_time_stamp:str=CURRENT_TIME_STAMP):
        """
        Initializes the AppConfiguration class.
        config_file_path: str
        By default it will accept default config file path.
        """
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            artifact_dir = os.path.join(
                self.training_pipeline_config.artifact_dir, DATA_INGESTION_ARTIFACT_DIR, self.time_stamp)

            data_ingestion_config = self.config_info[DATA_INGESTION_CONFIG_KEY]
            raw_data_dir = os.path.join(
                artifact_dir, data_ingestion_config[DATA_INGESTION_RAW_DATA_DIR_KEY])
            tgz_download_dir = os.path.join(
                artifact_dir, data_ingestion_config[DATA_INGESTION_TGZ_DOWNLOAD_DIR_KEY])

            ingested_dir_name = os.path.join(artifact_dir,
                                             data_ingestion_config[DATA_INGESTION_DIR_NAME_KEY])

            ingested_train_dir = os.path.join(ingested_dir_name,
                                              data_ingestion_config[DATA_INGESTION_TRAIN_DIR_KEY])

            ingested_test_dir = os.path.join(ingested_dir_name,
                                             data_ingestion_config[DATA_INGESTION_TEST_DIR_KEY])

            response = DataIngestionConfig(dataset_download_url=data_ingestion_config[DATA_INGESTION_DOWNLOAD_URL_KEY],
                                           raw_data_dir=raw_data_dir,
                                           tgz_download_dir=tgz_download_dir,
                                           ingested_train_dir=ingested_train_dir,
                                           ingested_test_dir=ingested_test_dir
                                           )
            logging.info(f"Data Ingestion Config: {response}")

            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_vaidation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            schema_file_path = os.path.join(
                ROOT_DIR, data_vaidation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            response = DataValidationConfig(schema_file_path=schema_file_path)
            logging.info(response)
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = os.path.join(
                self.training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_ARTIFACT_DIR, self.time_stamp)

            data_transormation_config = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            add_bedroom_per_room = data_transormation_config[
                DATA_TRANSFORMATION_ADD_BEDROOM_PER_ROOM_KEY]

            transformed_dir = os.path.join(
                artifact_dir, data_transormation_config[DATA_TRANSFORMATION_DIR_NAME_KEY])
            transformed_train_dir = os.path.join(
                transformed_dir, data_transormation_config[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY])
            transformed_test_dir = os.path.join(
                transformed_dir, data_transormation_config[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY])

            preprocessing_dir = os.path.join(
                artifact_dir, data_transormation_config[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY])

            preprocessed_file_name = data_transormation_config[
                DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]

            preprocessed_object_file_path = os.path.join(
                preprocessing_dir, preprocessed_file_name)

            response = DataTransformationConfig(add_bedroom_per_room=add_bedroom_per_room,
                                                transformed_test_dir=transformed_test_dir,
                                                transformed_train_dir=transformed_train_dir,
                                                preprocessed_object_file_path=preprocessed_object_file_path
                                                )

            logging.info(f"Data Transformation Config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_config = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_TRAINER_ARTIFACT_DIR,
                                        self.time_stamp)

            model_dir = os.path.join(artifact_dir, model_trainer_config[MODEL_TRAINER_DIR_KEY])
            model_file_path = os.path.join(model_dir, model_trainer_config[MODEL_TRAINER_FILE_NAME_KEY])

            base_accuracy = model_trainer_config[MODEL_TRAINER_BASE_ACCURACY_KEY]

            response = ModelTrainerConfig(trained_model_file_path=model_file_path,
                                          base_accuracy=base_accuracy,
                                          )
            logging.info(f"Model Trainer Config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = os.path.join(self.training_pipeline_config.artifact_dir,
                                        MODEL_EVALUATION_ARTIFACT_DIR, )

            model_evaluation_file_path = os.path.join(artifact_dir,
                                                      model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY])
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                             time_stamp=self.time_stamp)

            logging.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(ROOT_DIR, model_pusher_config[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)

            response = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(
                ROOT_DIR, training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            response = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training Pipeline Config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys) from e


    def get_housing_prediction_model_dir(self) -> str:
        try:
            model_pusher_config = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            return os.path.join(ROOT_DIR, model_pusher_config[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY])
        except Exception as e:
            raise AppException(e, sys) from e

if __name__ == '__main__':
    try:
        app_config = AppConfiguration()
        print(app_config.get_data_ingestion_config())
    except Exception as e:
        print(e)
