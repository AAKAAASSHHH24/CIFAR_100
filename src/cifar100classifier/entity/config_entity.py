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
   
    

    
    
