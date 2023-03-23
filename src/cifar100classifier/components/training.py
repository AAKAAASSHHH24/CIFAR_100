from cifar100classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cifar100classifier.entity import TrainingConfig
from cifar100classifier.constants import *
from pathlib import Path

import tensorflow as tf

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_training_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self,train_data_generator, valid_data_generator, callback_list: list):

        self.model.fit(
            train_data_generator,
            validation_data=valid_data_generator,
            epochs=self.config.params_epochs,
            verbose=1,
            callbacks=callback_list
        )
        

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )