import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Ensure logging is set up
LOG_FILE_PATH = os.path.join(os.getcwd(), 'logs', 'app.log')
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Configuration for Data Ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Check the paths for debugging
            print(f"Train Data Path: {self.ingestion_config.train_data_path}")
            print(f"Test Data Path: {self.ingestion_config.test_data_path}")
            print(f"Raw Data Path: {self.ingestion_config.raw_data_path}")

            # Ensure the path to the CSV file is correct
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as a dataframe')

            # Ensure the artifacts directory is created
            artifacts_dir = os.path.dirname(self.ingestion_config.train_data_path)
            print(f"Creating artifacts directory: {artifacts_dir}")
            os.makedirs(artifacts_dir, exist_ok=True)
            logging.info(f"Directory {artifacts_dir} created successfully")

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Perform the train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Create an instance of DataIngestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        # Create instances of DataTransformation and ModelTrainer
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")
