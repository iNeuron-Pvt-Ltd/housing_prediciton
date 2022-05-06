from app_logger import logging
from app_exception import AppException
from collections import namedtuple

from app_src.stage_01_data_validation import DataValidation
from app_src.stage_02_data_transformation import DataTransformation
from app_src.stage_05_model_pusher import ModelPusher


DataIngestionConfig = namedtuple("DatasetConfig", ["name", "path", "type"])


DataValidationConfig = namedtuple("DataValidationConfig", ["name", "path", "type"])


DataTransformationConfig = namedtuple("DataTransformationConfig", ["name", "path", "type"])


ModelTrainerConfig  = namedtuple("ModelTrainerConfig", ["name", "path", "type"])


ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["name", "path", "type"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["name", "path", "type"])




