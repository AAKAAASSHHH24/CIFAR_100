from cifar100classifier.components.prepare_callbacks import PrepareCallback
from cifar100classifier.components.data_generation import DataGenerator, Generator
from cifar100classifier.components.training import Training
from cifar100classifier.config import ConfigurationManager
from cifar100classifier import logger


STAGE_NAME = 'TRAINING STAGE'

def main():
    config = ConfigurationManager()
    
    data_generation_config = config.get_data_generation_config()
    data_generation = DataGenerator(config=data_generation_config)
    X_train_data, X_val_data, y_train_data, y_val_data= data_generation.get_train_val_data()
    category, subcategory = data_generation.get_metadata()
    train_data_generator = Generator(images= X_train_data, labels= y_train_data, augment=True, config=config.get_generator_config())
    valid_data_generator = Generator(images = X_val_data, labels=y_val_data, augment=False, config=config.get_generator_config()) # augmentation turned off for validation set during training
    
    callbacks_config = config.get_prepare_callback_config()
    prepare_callbacks = PrepareCallback(config=callbacks_config)
    callback_list = prepare_callbacks.get_callbacks()
    
    training_config = config.get_training_config()
    training = Training(config=training_config)
    training.get_training_model()
    training.train(train_data_generator=train_data_generator,
                   valid_data_generator=valid_data_generator,
                   callback_list=callback_list
    )
    

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
    except Exception as e:
        raise e