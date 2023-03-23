from cifar100classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cifar100classifier.entity import DataIngestionConfig, DataGenerationConfig, GeneratorConfig, PrepareBaseModelConfig,PrepareCallbacksConfig,TrainingConfig
from cifar100classifier.utils import read_yaml, create_directories

from pathlib import Path
import os


#CONFIGURATION MANAGER DEFINED

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
        
    def get_data_generation_config(self) -> DataGenerationConfig:
        config = self.config.data_generation
        
        create_directories([Path(config.root_dir), 
                            Path(config.transformed_data_path), 
                            Path(config.metadata_path)])

        data_generation_config = DataGenerationConfig(
            base_dir= config.base_dir,
            root_dir=config.root_dir,
            train_file=config.train_file,
            test_file=config.test_file,
            meta_file=config.meta_file,
            transformed_data_path= config.transformed_data_path,
            metadata_path= config.metadata_path
        )

        return data_generation_config

    def get_generator_config(self) -> GeneratorConfig:
        config = self.config.generator
        
        create_directories([config.root_dir])

        generator_config = GeneratorConfig(
            root_dir=Path(config.root_dir),
            transformed_data_path= config.transformed_data_path,
            metadata_path= config.metadata_path,
            mode= self.params.mode,
            height=self.params.height,
            width=self.params.width,
            channels=self.params.channels,
            n_classes=self.params.n_classes,
            input_shape=self.params.input_shape,
            dim=self.params.dim,
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            shuffle= self.params.shuffle
        )

        return generator_config
    
    def get_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
    
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.input_shape,
            params_learning_rate=self.params.learning_rate,
            params_include_top=self.params.include_top,
            params_weights=self.params.weights,
            n_classes=self.params.n_classes
        )

        return prepare_base_model_config
    
    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path(model_ckpt_dir),
            Path(config.tensorboard_root_log_dir)
        ])

        callback_config = PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath)
        )

        return callback_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_training_model = self.config.prepare_base_model
        params = self.params
        
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_training_model.updated_base_model_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE
        )

        return training_config