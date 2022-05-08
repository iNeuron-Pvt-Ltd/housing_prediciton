from cmath import log
from app_logger import logging
from app_exception import AppException
from app_entity import ModelTrainerConfig, ModelTrainerArtifact
from sklearn.linear_model import LinearRegression
from app_entity import DataTransformationArtifact, MetricInfoArtifact
from app_src import DataTransformation
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys
from app_util import load_object, save_object
from sklearn.tree import DecisionTreeRegressor


class TrainedModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_trainer_config, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'=' * 20}Model trainer log started.{'=' * 20} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_list = ModelTrainer.get_list_of_models()
        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def get_list_of_models():
        try:
            model_list = []
            linear_regression = LinearRegression()
            model_list.append(linear_regression)
            decision_tree_regressor = DecisionTreeRegressor()
            model_list.append(decision_tree_regressor)
            return model_list
        except Exception as e:
            raise AppException(e, sys) from e

    def fit(self, X, y):
        try:
            for model in self.model_list:
                logging.info(
                    f"Started training model: [{type(model).__name__}]")
                model.fit(X, y)
                logging.info(
                    f"Finished training model: [{type(model).__name__}]")

        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def evaluate_model(model_list: list, X_train, y_train, X_test, y_test, base_accuracy=0.5) -> MetricInfoArtifact:
        try:
            index_number = 0
            metric_info_artifact = None
            for model in model_list:
                model_name = str(model)
                logging.info(
                    f"Started evaluating model: [{type(model).__name__}]")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_acc = r2_score(y_train, y_train_pred)
                test_acc = r2_score(y_test, y_test_pred)
                train_rmse = mean_squared_error(y_train, y_train_pred)
                test_rmse = mean_squared_error(y_test, y_test_pred)

                # Calculating harmonic mean of train_accuracy and test_accuracy
                model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
                diff_test_train_acc = abs(test_acc - train_acc)
                message = f"{'*' * 20}{model_name} metric info{'*' * 20}"
                logging.info(f"{message}")
                message = f"\n\t\tTrain accuracy: [{train_acc}]."
                message += f"\n\t\tTest accuracy: [{test_acc}]."
                message += f"\n\t\tTrain rmse: [{train_rmse}]."
                message += f"\n\t\tTest rmse: [{test_rmse}]."
                message += f"\n\t\tModel accuracy: [{model_accuracy}]."
                message += f"\n\t\tBase accuracy: [{base_accuracy}]."
                message += f"\n\t\tDiff test train accuracy: [{diff_test_train_acc}]."
                logging.info(message)
                message = f"{'*' * 20}{model_name} metric info{'*' * 20}"
                logging.info(message)

                if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                    base_accuracy = model_accuracy
                    metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                              model_object=model,
                                                              train_rmse=train_rmse,
                                                              test_rmse=test_rmse,
                                                              train_accuracy=train_acc,
                                                              test_accuracy=test_acc,
                                                              model_accuracy=model_accuracy,
                                                              index_number=index_number)

                    logging.info(
                        f"Acceptable model found {metric_info_artifact}. ")
                index_number += 1

            if metric_info_artifact is None:
                logging.info(
                    f"No model found with higher accuracy than base accuracy")

            return metric_info_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            train_dataset = DataTransformation.load_numpy_array_data(
                file_path=train_file_path)
            test_dataset = DataTransformation.load_numpy_array_data(
                file_path=test_file_path)

            X_train, y_train = train_dataset[:, :-1], train_dataset[:, -1]
            X_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]

            self.fit(X_train, y_train)
            model_metric_artifact = ModelTrainer.evaluate_model(model_list=self.model_list,
                                                                X_train=X_train,
                                                                y_train=y_train,
                                                                X_test=X_test,
                                                                y_test=y_test,
                                                                base_accuracy=self.model_trainer_config.base_accuracy)

            if model_metric_artifact is None:
                raise Exception("None of suggested model is able to achieve least base accuracy")
            preprocessed_object_file_path = self.data_transformation_artifact.preprocessed_object_file_path

            preprocessed_object = load_object(
                file_path=preprocessed_object_file_path)

            trained_model = TrainedModel(
                preprocessing_object=preprocessed_object,
                trained_model_object=model_metric_artifact.model_object)

            trained_model_path = self.model_trainer_config.trained_model_file_path
            logging.info(f"Saving trained model to: {trained_model_path}")
            save_object(file_path=trained_model_path, obj=trained_model)
            logging.info(f"Saved trained model to: {trained_model_path}")

            response = ModelTrainerArtifact(is_trained=True,
                                            message="Model trained successfully",
                                            trained_model_file_path=trained_model_path,
                                            train_rmse=model_metric_artifact.train_rmse,
                                            test_rmse=model_metric_artifact.test_rmse,
                                            train_accuracy=model_metric_artifact.train_accuracy,
                                            test_accuracy=model_metric_artifact.test_accuracy,
                                            model_accuracy=model_metric_artifact.model_accuracy
                                            )
            logging.info(f"Trained model artifact: {response}.")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model trainer log completed.{'=' * 20} ")
