from cifar100classifier.entity import DataIngestionConfig
from cifar100classifier import logger
from cifar100classifier.utils import *
# CREATE COMPONENT OF DATA INGESTION

import os
import urllib.request as request
import tqdm

# importing the "tarfile" module
import tarfile



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        logger.info("Trying to download file...")
        if not os.path.exists(self.config.local_data_file):
            logger.info("Download started...")
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
            
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  



    def unzip_and_clean(self):
        logger.info(f"extracting from compressed file and removing unawanted files")
        working_dir =self.config.unzip_dir
        os.chdir(working_dir)
        zf= tarfile.open('data.tar.gz')
        list_of_files = zf.getnames()
        for f in list_of_files:
            if not f.endswith(".txt~"):
                zf.extract(f)
                    
        zf.close()
                    