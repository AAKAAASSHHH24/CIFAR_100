import numpy as np
import albumentations as albu
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

from cifar100classifier.constants import *
from cifar100classifier.utils import read_yaml, create_directories, unpickle
from cifar100classifier.entity import DataGenerationConfig





class DataGenerator:
   def __init__(self, config: DataGenerationConfig, params_filepath = PARAMS_FILE_PATH ):
      self.config = config
      self.params = read_yaml(params_filepath)
         
   #function to open the files in the Python version of the dataset
   def get_data(self):
      train_data = unpickle(Path(self.config.train_file))
      test_data = unpickle(Path(self.config.test_file))
      
      #4D array input for building the CNN model using Keras
      X_train = train_data['data']
      X_train = X_train.reshape(len(X_train),3,32,32).transpose(0,2,3,1)
      
      #transforming the testing dataset
      X_test = test_data['data']
      X_test = X_test.reshape(len(X_test),3,32,32).transpose(0,2,3,1)
      
      y_train = train_data['fine_labels']
      y_train = to_categorical(y_train, self.params.n_classes)

      y_test = test_data['fine_labels']
      y_test = to_categorical(y_test, self.params.n_classes)
      
      return X_train, X_test, y_train, y_test
   
   def get_train_val_data(self):
      sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=123)
      
      X_train, X_test, y_train, y_test = self.get_data()

      for train_index, val_index in sss.split(X_train, y_train):
         X_train_data, X_val_data = X_train[train_index], X_train[val_index]
         y_train_data, y_val_data = y_train[train_index], y_train[val_index]
      
      return X_train_data, X_val_data, y_train_data, y_val_data
   
   def get_metadata(self): 
      
      meta_data = unpickle(Path(self.config.meta_file)) 
      #storing coarse labels along with its number code in a dataframe
      category = pd.DataFrame(meta_data['coarse_label_names'], columns=['SuperClass'])
      
      #storing fine labels along with its number code in a dataframe
      subcategory = pd.DataFrame(meta_data['fine_label_names'], columns=['SubClass'])
      
      return category, subcategory
   