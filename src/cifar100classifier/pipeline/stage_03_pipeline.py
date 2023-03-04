from cifar100classifier.components.prepare_base_model import PrepareBaseModel
from cifar100classifier.config import ConfigurationManager
from cifar100classifier import logger


STAGE_NAME = 'BASE MODEL PREPARATION STAGE'

def main():
    config = ConfigurationManager()
    prepare_base_model_config = config.get_base_model_config()
    prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
    prepare_base_model.get_base_model()
    prepare_base_model.update_base_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
    except Exception as e:
        raise e