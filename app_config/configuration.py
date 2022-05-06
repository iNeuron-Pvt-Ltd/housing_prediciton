from app_logger import logging
from app_exception import AppException
import os,sys

ROOT_DIR = os.getcwd()
#Varibale declaration
#Data Ingestion related variables
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_download_url"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"


#Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"


#Data Transformation related variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"

# Model Training related variables

MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"

# Model Evaluation related variables

MODEL_EVALUATION_CONFIG_KEY = "model_evaluation_config"


#Model Pusher config key
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"



#Training pipeline realted variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"






class AppConfiguration:
    pass




