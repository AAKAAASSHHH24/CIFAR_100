from cifar100classifier.components.data_generation import DataGenerator
from cifar100classifier.config import ConfigurationManager
from cifar100classifier import logger


STAGE_NAME = 'DATA GENERATION STAGE'

def main():
    config = ConfigurationManager()
    data_generation_config = config.get_data_generation_config()
    data_generation = DataGenerator(config=data_generation_config)
    data_generation.get_train_val_data()
    data_generation.get_metadata()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
    except Exception as e:
        raise e