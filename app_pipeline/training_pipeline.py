from app_logger import logging
from app_exception import AppException
from app_config import AppConfiguration
import os
import sys
from app_entity import DataIngestionArtifact, ModelTrainerArtifact, DataValidationArtifact, DataTransformationArtifact, \
    DataTransformationConfig, ModelEvaluationArtifact, ModelPusherArtifact
from app_src import DataIngestion, ModelPusher
from app_src import DataValidation, DataTransformation, ModelTrainer, ModelEvaluation


class TrainingPipeline:

    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        """
        TrainingPipeline constructor
        app_config: AppConfiguration

        """
        try:
            self.app_config = app_config

        except Exception as e:
            raise AppException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Starts data ingestion and 
        """
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.app_config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise AppException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Starts data validation
        data_ingestion_artifact: DataIngestionArtifact
        """
        try:
            data_validation = DataValidation(
                data_validation_config=self.app_config.get_data_validation_config(),
                data_ingestion_artifact=data_ingestion_artifact)

            return data_validation.initiate_data_validation()
        except Exception as e:
            raise AppException(e, sys) from e

    def start_data_transformation(self,
                                  data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact,
                                  ) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_transformation_config=self.app_config.get_data_transformation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise AppException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.app_config.get_model_trainer_config(),
                data_transformation_artifact=data_transformation_artifact,

            )
            return model_trainer.initiate_model_trainer()
        except Exception as e:
            raise AppException(e, sys) from e

    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        try:
            model_pusher = ModelPusher(
                model_pusher_config=self.app_config.get_model_pusher_config(),
                model_evaluation_artifact=model_eval_artifact
            )
            return model_pusher.initiate_model_pusher()
        except Exception as e:
            raise AppException(e, sys) from e

    def start_model_evaluation(
            self, data_ingestion_artifact: DataIngestionArtifact,
            data_validation_artifact: DataValidationArtifact,
            model_trainer_artifact: ModelTrainerArtifact)->ModelEvaluationArtifact:
        try:
            model_eval = ModelEvaluation(
                model_evaluation_config=self.app_config.get_model_evaluation_config(),
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact)
            return model_eval.initiate_model_evaluation()
        except Exception as e:
            raise AppException(e, sys) from e

    def start_training_pipeline(self):
        try:
            logging.info("Starting training pipeline")
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact)

            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_eval_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Training pipeline completed")

        except Exception as e:
            raise AppException(e, sys) from e
