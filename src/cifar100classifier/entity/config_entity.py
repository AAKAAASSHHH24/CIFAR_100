from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class DataGenerationConfig:
    base_dir: Path
    root_dir: Path
    train_file: Path
    test_file: Path
    meta_file: Path
    transformed_data_path: Path
    metadata_path: Path
    
@dataclass(frozen=True)
class GeneratorConfig:
    root_dir: Path
    transformed_data_path: Path
    metadata_path: Path
    mode: str
    height: int
    width: int
    channels: int
    n_classes: int
    input_shape: tuple
    dim: tuple
    epochs: int
    batch_size: int
    shuffle: bool
    

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: tuple
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    n_classes: int
   
   
@dataclass(frozen=True)
class PrepareCallbacksConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path
    
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    params_epochs: int
    params_batch_size: int

   
    

    
    
