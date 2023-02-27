from cifar100classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cifar100classifier.entity import DataIngestionConfig, DataGenerationConfig
from cifar100classifier.utils import read_yaml, create_directories



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
        
        create_directories([config.root_dir])

        data_generation_config = DataGenerationConfig(
            root_dir=config.root_dir,
            train_file=config.train_file,
            test_file=config.test_file,
            meta_file=config.meta_file 
        )

        return data_generation_config