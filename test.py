from app_logger import logging
from app_pipeline import TrainingPipeline
from app_entity.model_factory import ModelFactory,get_sample_model_config_yaml_file
import os
from app_src.stage_02_data_transformation import DataTransformation

if __name__ == '__main__':
    try:
        TrainingPipeline().start_training_pipeline()
        #get_sample_model_config_yaml_file("logs")
        # model_config_file = os.path.join("config", "model.yaml")
        # model_factory = ModelFactory(model_config_path=model_config_file)
        # file_path = r'D:\Project\housing_prediciton\housing\artifact\data_transformation\2022-06-30-16-53-38\transformed_data\train\housing_transformed.npy'
        # dataset = DataTransformation.load_numpy_array_data(file_path=file_path)
        # X, y = dataset[:, :-1], dataset[:, -1]
        # best_model = model_factory.get_best_model(X, y)
        # print(best_model)
    except Exception as e:
        logging.info(e)
