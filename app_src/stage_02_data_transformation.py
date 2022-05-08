from app_logger import logging
from app_exception import AppException
from app_entity import DataTransformationConfig, DataIngestionArtifact, DataValidationArtifact
import os
import sys
from app_util import read_yaml_file
from sklearn.preprocessing import LabelBinarizer
from app_config import DATASET_SCHEMA_COLUMNS_KEY, DATASET_SCHEMA_TARGET_COLUMN_KEY, DATASET_SCHEMA_DOMAIN_VALUE_KEY
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from app_entity import DataTransformationArtifact
import dill
from app_util import save_object

COLUMN_TOTAL_ROOMS = "total_rooms"
COLUMN_POPULATION = "population"
COLUMN_HOUSEHOLDS = "households"
COLUMN_TOTAL_BEDROOM = "total_bedrooms"


#   longitude: float
#   latitude: float
#   housing_median_age: float
#   total_rooms: float
#   total_bedrooms: float
#   population: float
#   households: float
#   median_income: float
#   median_house_value: float
#   ocean_proximity: category
#   income_cat: float


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True,
                 total_rooms_ix=3,
                 population_ix=5,
                 households_ix=6,
                 total_bedrooms_ix=4, columns=None):
        """
        FeatureGenerator Initialization
        add_bedrooms_per_room: bool
        total_rooms_ix: int index number of total rooms columns
        population_ix: int index number of total population columns
        households_ix: int index number of  households columns
        total_bedrooms_ix: int index number of bedrooms columns
        """
        try:
            self.columns = columns
            if self.columns is not None:
                total_rooms_ix = self.columns.index(COLUMN_TOTAL_ROOMS)
                population_ix = self.columns.index(COLUMN_POPULATION)
                households_ix = self.columns.index(COLUMN_HOUSEHOLDS)
                total_bedrooms_ix = self.columns.index(COLUMN_TOTAL_BEDROOM)

            self.add_bedrooms_per_room = add_bedrooms_per_room
            self.total_rooms_ix = total_rooms_ix
            self.population_ix = population_ix
            self.households_ix = households_ix
            self.total_bedrooms_ix = total_bedrooms_ix
        except Exception as e:
            raise AppException(e, sys) from e

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            room_per_household = X[:, self.total_rooms_ix] / \
                                 X[:, self.households_ix]
            population_per_household = X[:, self.population_ix] / \
                                       X[:, self.households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, self.total_bedrooms_ix] / \
                                    X[:, self.total_rooms_ix]
                generated_feature = np.c_[
                    X, room_per_household, population_per_household, bedrooms_per_room]
            else:
                generated_feature = np.c_[
                    X, room_per_household, population_per_household]

            return generated_feature
        except Exception as e:
            raise AppException(e, sys) from e


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        """
        data_transformation_config: DataTransformationConfig
        data_ingestion_artifact: DataIngestionArtifact
        data_validation_artifact: DataValidationArtifact
        
        """
        try:
            logging.info(f"{'=' * 20}Data Transformation log started.{'=' * 20} ")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
        try:
            # reading the dataset schema file
            datatset_schema = read_yaml_file(schema_file_path)

            # extracting the columns info from the schema file
            schema = datatset_schema[DATASET_SCHEMA_COLUMNS_KEY]

            # reading the dataset
            dataframe = pd.read_csv(file_path)
            error_messgae = ""
            for column in dataframe.columns:
                if column in list(schema.keys()):
                    dataframe[column].astype(schema[column])
                else:
                    error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
            if len(error_messgae) > 0:
                raise Exception(error_messgae)
            return dataframe
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_transformer(self) -> ColumnTransformer:
        try:

            # reading schema file path
            schema_file_path = self.data_validation_artifact.schema_file_path
            data_transformation_config = self.data_transformation_config
            dataset_schema = read_yaml_file(schema_file_path)

            # spliting input columns and target column
            # TARGET_COLUMN
            target_column = dataset_schema[DATASET_SCHEMA_TARGET_COLUMN_KEY]

            # dropping categorical columns selecting only numerical columns to replace missing
            # value of numerical column
            columns = list(dataset_schema[DATASET_SCHEMA_COLUMNS_KEY].keys())
            columns.remove(target_column)

            categorical_column = []
            for column_name, data_type in dataset_schema[DATASET_SCHEMA_COLUMNS_KEY].items():
                if data_type == "category" and column_name != target_column:
                    categorical_column.append(column_name)

            numerical_column = list(
                filter(lambda x: x not in categorical_column, columns))

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('feature_generator', FeatureGenerator(
                    add_bedrooms_per_room=data_transformation_config.add_bedroom_per_room,
                    columns=numerical_column
                )),
                ('scaler', StandardScaler())
            ]
            )

            cat_pipeline = Pipeline(steps=[
                 ('impute', SimpleImputer(strategy="most_frequent")),
                 ('one_hot_encoder', OneHotEncoder()),
                 ('scaler', StandardScaler(with_mean=False))
            ]
            )
            logging.info(f"Categorical columns: {categorical_column}")
            logging.info(f"Numerical columns: {numerical_column}")
            preprocessing = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_column),
                ('cat_pipeline', cat_pipeline, categorical_column),
            ])
            return preprocessing

        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def save_numpy_array_data(file_path: str, array: np.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def load_numpy_array_data(file_path: str) -> np.array:
        """
        load numpy array data from file
        file_path: str location of file to load
        return: np.array data loaded
        """
        try:
            with open(file_path, 'rb') as file_obj:
                
                return np.load(file_obj)
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"train file path: [{train_file_path}]\n \
            test file path: [{test_file_path}]\n \
            schema_file_path: [{schema_file_path}]\n. ")

            # loading the dataset
            logging.info(f"Loading train and test dataset...")
            train_dataframe = DataTransformation.load_data(file_path=train_file_path,
                                                           schema_file_path=schema_file_path
                                                           )

            test_dataframe = DataTransformation.load_data(file_path=test_file_path,
                                                          schema_file_path=schema_file_path)

            logging.info("Data loaded successfully.")

            target_column_name = read_yaml_file(file_path=schema_file_path)[DATASET_SCHEMA_TARGET_COLUMN_KEY]
            logging.info(f"Target column name: [{target_column_name}].")

            # target_column
            logging.info(f"Converting target column into numpy array.")
            train_target_arr = np.array(train_dataframe[target_column_name])
            test_target_arr = np.array(test_dataframe[target_column_name])
            logging.info(f"Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info(f"Dropping target column from the dataframe.")
            train_dataframe.drop(target_column_name, axis=1, inplace=True)
            test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info(f"Dropping target column from the dataframe completed.")

            logging.info(f"Creating preprocessing object.")
            preprocessing = self.get_data_transformer()
            logging.info(f"Creating preprocessing object completed.")
            logging.info(f"Preprocessing object learning started on training dataset.")
            logging.info(f"Transformation started on training dataset.")
            train_input_arr = preprocessing.fit_transform(train_dataframe)
            logging.info(f"Preprocessing object learning completed on training dataset.")

            logging.info(f"Transformation started on testing dataset.")
            test_input_arr = preprocessing.transform(test_dataframe)
            logging.info(f"Transformation completed on testing dataset.")

            # adding target column back to the numpy array
            logging.info("Started concatenation of target column back  into transformed numpy array.")
            train_arr = np.c_[train_input_arr, train_target_arr]
            test_arr = np.c_[test_input_arr, test_target_arr]
            logging.info("Completed concatenation of  target column back  into transformed numpy array.")

            # generating file name such as housing_transformed.npy
            file_name = os.path.basename(train_file_path)
            file_extension_starting_index = file_name.find(".")
            file_name = file_name[:file_extension_starting_index]
            file_name = file_name + "_transformed.npy"
            logging.info(f"File name: [{file_name}] for transformed dataset.")

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            transformed_train_file_path = os.path.join(transformed_train_dir, file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, file_name)
            logging.info(f"Transformed train file path: [{transformed_train_file_path}].")
            logging.info(f"Transformed test file path: [{transformed_test_file_path}].")
            # saving the transformed data 
            logging.info(f"Saving transformed train and test dataset to file.")
            DataTransformation.save_numpy_array_data(file_path=transformed_train_file_path,
                                                     array=train_arr)

            DataTransformation.save_numpy_array_data(file_path=transformed_test_file_path,
                                                     array=test_arr)
            logging.info(f"Saving transformed train and test dataset to file completed.")

            logging.info(f"Saving preprocessing object")
            preprocessed_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            # saving the preprocessed object
            save_object(file_path=preprocessed_object_file_path,
                        obj=preprocessing)
            logging.info(f"Saving preprocessing object in file: [{preprocessed_object_file_path}] completed.")
            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformed successfully",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessed_object_file_path)
            logging.info(f"Data Transformation artifact: [{data_transformation_artifact}] created successfully")
            return data_transformation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
    
        logging.info(f"{'=' * 20}Data Transformation log ended.{'=' * 20} ")
