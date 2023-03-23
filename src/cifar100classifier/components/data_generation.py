import numpy as np
import os
import albumentations as albu
import pandas as pd
import cv2
import keras
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

from cifar100classifier.constants import *
from cifar100classifier.utils import read_yaml, create_directories, unpickle, get_current_time_stamp
from cifar100classifier.entity import DataGenerationConfig, GeneratorConfig
from cifar100classifier import logger
  
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
      
      train_val_dir = os.path.join(self.config.transformed_data_path, get_current_time_stamp())
      os.makedirs(train_val_dir,exist_ok=True)
      
      current_dir = os.getcwd()
      os.chdir(train_val_dir)
      np.save('X_train_data', X_train_data)
      np.save('X_val_data', X_val_data)
      np.save('y_train_data', y_train_data)
      np.save('y_val_data', y_val_data)
      
      logger.info(f"{train_val_dir} created and loaded successfully")
      os.chdir(current_dir)
      
      return X_train_data, X_val_data, y_train_data, y_val_data
   
   def get_metadata(self): 
      
      meta_data = unpickle(Path(self.config.meta_file)) 
      #storing coarse labels along with its number code in a dataframe
      category = pd.DataFrame(meta_data['coarse_label_names'], columns=['SuperClass'])
      
      #storing fine labels along with its number code in a dataframe
      subcategory = pd.DataFrame(meta_data['fine_label_names'], columns=['SubClass'])
      
      metadata_dir = os.path.join(self.config.metadata_path, get_current_time_stamp())
      os.makedirs(metadata_dir,exist_ok=True)
      
      current_dir = os.getcwd()
      os.chdir(metadata_dir)
      np.save('category', category)
      np.save('subcategory', subcategory)
      logger.info(f"{metadata_dir} created and loaded successfully")
      os.chdir(current_dir)
      
      return category, subcategory
   
   
   
class Generator(keras.utils.Sequence): 
    
    """ defined as a subclass of Keras Sequence class, which provides efficient multi-threaded data loading and processing"""
    
    def __init__(self, config: GeneratorConfig, images, labels=None, augment=False):
        
        #initializing the configuration of the generator
        self.images = images
        self.labels = labels
        self.augment = augment
        self.config = config
        self.on_epoch_end()
        
   
    # method to be called after every epoch
    def on_epoch_end(self):
        
        """This method generates an array of indexes based on the number of images in the 
        dataset and shuffles them if shuffle is set to True."""
        
        self.indexes = np.arange(self.images.shape[0])
        #if self.config.shuffle == True:
        np.random.shuffle(self.indexes)
    
    #return numbers of steps in an epoch using samples and batch size
    def __len__(self):
        
        """Returns the number of steps in an epoch, which is calculated as the 
        total number of samples divided by the batch size, rounded down to the nearest integer."""
        
        return int(np.floor(len(self.images) / self.config.batch_size))
    
     #this method is called with the batch number as an argument to obtain a given batch of data
    
    def resize_img(img, shape):
        return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) 
    
    def __getitem__(self, index):
        
        """This method is called with the batch number as an argument to obtain a given batch of data. 
        It generates a mini-batch of X and y, where X is an array of preprocessed images 
        and y is an array of corresponding labels (if mode is fit). 
        If augment is set to True, data augmentation is applied to the training dataset. 
        If mode is set to predict, only X is returned."""
        
        #generate indexes of batch
        batch_indexes = self.indexes[index * self.config.batch_size:(index+1) * self.config.batch_size]
        
        #generate mini-batch of X
        X = np.empty((self.config.batch_size, *self.config.dim, self.config.channels))
        
        for i, ID in enumerate(batch_indexes):
            #generate pre-processed image
            img = self.images[ID]
            #image rescaling
            img = img.astype(np.float32)/255.
            #resizing as per new dimensions
            img = self.resize_img(img, self.config.dim)   # use the resize function
            X[i] = img
            
        #generate mini-batch of y
        if self.config.mode == 'fit':
            y = self.config.labels[batch_indexes]
            
            #augmentation on the training dataset
            if self.config.augment == True:
                X = self.__augment_batch(X)
            return X, y
        
        elif self.config.mode == 'predict':
            return X
        
        else:
            raise AttributeError("The mode should be set to either 'fit' or 'predict'.")
        
    
    #augmentation for one image
    def __random_transform(self, img):
        
        """This method applies a random transformation to a single image using the albumentations library, 
        which is a popular library for image augmentation."""
        
        composition = albu.Compose([albu.HorizontalFlip(p=0.5),
                                   albu.VerticalFlip(p=0.5),
                                   albu.GridDistortion(p=0.2),
                                   albu.ElasticTransform(p=0.2)])
        return composition(image=img)['image']
    
    #augmentation for batch of images
    def __augment_batch(self, img_batch):
        
        """This method applies a random transformation to a batch of images by calling 
        the __random_transform method for each image in the batch."""
        
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
        return img_batch
    