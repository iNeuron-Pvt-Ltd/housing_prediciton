from app_logger import logging
from app_pipeline import TrainingPipeline

if __name__ == '__main__':
    try:
        TrainingPipeline().start_training_pipeline()
    except Exception as e:
        logging.info(e)
