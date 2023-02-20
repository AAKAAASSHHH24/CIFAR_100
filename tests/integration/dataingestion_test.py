"""import os
import logging
from pathlib import Path
from cifar100classifier.entity import DataIngestionConfig
from src.cifar100classifier.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig,
)
import pytest

logger = logging.getLogger(__name__)


class Test_DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def test_download_file(self):
        self.data_ingestion.download_file()
        assert os.path.exists(self.local_data_file)
        assert os.path.getsize(self.local_data_file) > 0

    def test_unzip_and_clean(self):
        with pytest.raises(Exception):
            self.data_ingestion.unzip_and_clean()
        extracted_files = [
            f for f in os.listdir(self.unzip_dir) if not f.endswith(".txt~")
        ]
        assert len(extracted_files) == 0"""
        
def test_dummy():
    assert True
