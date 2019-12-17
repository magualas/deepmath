"""

generator.py

Generator for batch training on keras models


"""

import pandas as pd
import os
import numpy as np
import boto3
import tensorflow as tf
# print(tf.__version__)

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from botocore.client import ClientError
# from smart_open import smart_open
import csv


# config
BUCKET_NAME = 'sagemaker-cs281'
config = {'AWS_REGION':'us-east-2',        
          'S3_ENDPOINT':'s3.us-east-2.amazonaws.com', 
          'S3_USE_HTTPS':'1',                 
          'S3_VERIFY_SSL':'1',  }
os.environ.update(config)
line_counts = {'train':376968, 'test':122928, 'valid':104054}


class Keras_DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras
    
        Usage: 
            training_generator = generator.My_DataGenerator(dataset='train')
            validation_generator = generator.My_DataGenerator(dataset='valid')
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          verbose=1, use_multiprocessing=False,
                                          epochs=n_epochs)
                   
        Data is stored in three folders in S3 key 'deephol-data-processed/proofs/human'
            * /train
            * /valid
            * /test
        Files are have three path format. For all three folders, we keep the name X or Y_train
            * /X_train_{}.csv
            * /X_train_hyp_{}.csv
            * /Y_train.csv

    """
    def __init__(self, dataset='train',batch_size=64,
                 w_hyp=False, n_channels=1, n_classes=41, shuffle=True):
        self.w_hyp = w_hyp
        self.dim = 3000 if self.w_hyp else 1000 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        
        # paths
        X_paths, Y_path = self.get_partition_and_labels()
        self.features_keys_lst = X_paths
        self.label_key = Y_path[0]
        self.n = line_counts[self.dataset]
        self.partition_index = 0
        
        # initialize readers
        self.on_epoch_end()
        print('Generating examples from a set of {} examples'.format(self.n))

    def __len__(self):
        """ Denotes the number of batches per epoch 
            subtract 1 unfull batch per partition """
        
        return int(np.floor(self.n / self.batch_size)) - len(self.features_keys_lst) - 1

    def __getitem__(self, index):
        'Generate one batch of data'
        if self.partition_index >= len(self.features_keys_lst) - 1:
#             pass #if you put this pass on the 
            self.on_epoch_end()
        
        try:
            X, y = next(self.reader_X_lst[self.partition_index]), next(self.reader_Y)
        except Exception as e:
            self.partition_index += 1
            X, y = next(self.reader_X_lst[self.partition_index]), next(self.reader_Y)
        else:
            if len(X) < 64:
                self.partition_index += 1
                X, y = next(self.reader_X_lst[self.partition_index]), next(self.reader_Y)
        
        return X.values, y.values
    
    def _initialize_readers(self):
        paths_X = [os.path.join('s3://', BUCKET_NAME, x) for x in self.features_keys_lst]
        path_Y = os.path.join('s3://', BUCKET_NAME, self.label_key)
        self.reader_X_lst = [pd.read_csv(path, chunksize=self.batch_size, header=None, engine='python')
                             for path in paths_X]
        self.reader_Y = pd.read_csv(path_Y, chunksize=self.batch_size, header=None, engine='python')

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # re initialize readers
        self._initialize_readers()
        self.list_partitions = self.features_keys_lst
        if self.shuffle == True:
            np.random.shuffle(self.list_partitions)
        
        # start from begining
        self.partition_index = 0
    
    def get_partition_and_labels(self):
        """ Create a dictionary called partition where:
            - in partition['train']: a list of training IDs
            - in partition['validation']: a list of validation IDs
        """
        s3_r = boto3.resource('s3')
        my_bucket = s3_r.Bucket(BUCKET_NAME)
        full_dataset_key = 'deephol-data-processed/proofs/human'
        
        # paths as strings
        dataset_keys = {s: '{}/{}/'.format(full_dataset_key, s)
                        for s in ['train', 'test', 'valid']}
        partition = {dataset: [x.key for x in my_bucket.objects.filter(Prefix=dataset_keys[self.dataset])]
                     for dataset in ['train', 'test', 'valid']}
        print('Retrieving data from {}'.format(dataset_keys[self.dataset]))
        
        # get each file key
        y_file = [x for x in partition[self.dataset] if x.find('/Y_train') != (-1)]
        X_files_hyp = [x for x in partition[self.dataset] if x.find('/X_train_hyp_') != (-1)]
        X_files = X_files_hyp if self.w_hyp else set(partition[self.dataset]) - set(y_file) - set(X_files_hyp)
        
        # sort (will be shuffled if shuffle=True)
        X_files = sorted(X_files, key=lambda x: (len(x), x))
        
        return X_files, y_file
    
    
# tests


