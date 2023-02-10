from cifar100classifier.components.data_ingestion import DataIngestion
from cifar100classifier.config import ConfigurationManager
from cifar100classifier import logger


STAGE_NAME = 'DATA INGESTION STAGE'

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.unzip_and_clean()






if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
    except Exception as e:
        raise e
        