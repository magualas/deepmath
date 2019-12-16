"""

generator.py

Generator for batch training on keras models


"""

import os
import numpy as np
import pandas as pd
import csv
import random
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


# config
BUCKET_NAME = 'sagemaker-cs281'
config = {'AWS_REGION':'us-east-2',        
          'S3_ENDPOINT':'s3.us-east-2.amazonaws.com', 
          'S3_USE_HTTPS':'1',                 
          'S3_VERIFY_SSL':'1'}
os.environ.update(config)
line_counts = {'train': 376968, 'test': 122928, 'valid': 104054,
               'train_new': 376968, 'test_new': 122928, 'valid_new': 104054}


class Keras_DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras
    
        !! data_dir arg is not necessary atm
    
        Usage: 
            training_generator = generator.My_DataGenerator(data_dir='', dataset='train_new', 
                                                  w_hyp=W_HYP, batch_size=BATCH_SIZE,
                                                  shuffle=shuffle)
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
    def __init__(self, data_dir, dataset, batch_size=64, w_hyp=False, 
                 n_classes=41, shuffle=False, subset_frac=None):
        # input valid
        if dataset not in ['train_new', 'test_new', 'valid_new']:
            raise Exception('given dataset type must be _new to include the new dataset partitioning')
            
        # main attributes
        self.w_hyp = w_hyp
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = dataset
        self.data_dir = 'deephol-data-processed/proofs/human'
        self.subset_frac = subset_frac
    
        # batch and partitions ids (some training examples ommitted to have batches of equal size)
        self.batch_index = 0
        self.partition_size = 4096
        self.batches_per_partition = self.partition_size/self.batch_size
        n_examples = line_counts[self.dataset]
        end_to_skip = n_examples % 4096
        self.n = n_examples - end_to_skip
        self.batch_ids = list(range(int(self.n / self.batch_size)))
        print('# of batches: ', self.n / self.batch_size)
        
        # subset of data
        if self.subset_frac:
            self.n = self.n * subset_frac
            n_batches = int(len(self.batch_ids) * subset_frac)
            self.batch_ids = random.sample(self.batch_ids, n_batches)
            print('# of batches reduced to: ', self.n / self.batch_size)
        
#         # paths
#         X_paths, Y_paths = self._get_partition_and_labels()
#         self.features_keys = X_paths
#         self.labels_keys = Y_paths
        
        # randomize order if needed in epoch ends
        self.on_epoch_end()
        print('Generating examples from a set of {} examples \n'.format(self.n))

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return 500 #int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data """
        # find corresponding partition
        current_batch_id = self.batch_ids[index]
        current_partition = int(np.floor(current_batch_id / self.batches_per_partition))
        partition_path = os.path.join('s3://', BUCKET_NAME, self.data_dir, self.dataset)
        X_filename = 'X_train{}_{}.csv'.format('_hyp' if self.w_hyp else '', current_partition)
        Y_filename = 'Y_train_{}.csv'.format(current_partition)
        
        # get batch
        try:
            X_batch = pd.read_csv(os.path.join(partition_path, X_filename),
                                  skiprows=current_batch_id%self.partition_size, 
                                  nrows=self.batch_size,
                                  header=None)
            Y_batch = pd.read_csv(os.path.join(partition_path, Y_filename),
                                  skiprows=current_batch_id%self.partition_size, 
                                  nrows=self.batch_size,
                                  header=None)
        except Exception as e:
            print('ERROR: ', e)
            
        # needed to keep the .__next__() method working
        self.batch_index += 1
        
        return X_batch.values, Y_batch.values
    
    def __next__(self):
        """ Make whole object a generator"""
        return self.__getitem__(self.batch_index)
    
    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        # start from begining
        self.batch_index = 0

        if self.shuffle == True:
            np.random.shuffle(self.batch_ids)
    
#     def _get_partition_and_labels(self):
#         """ get all s3 paths to get data """
#         s3_r = boto3.resource('s3')
#         my_bucket = s3_r.Bucket(BUCKET_NAME)
        
#         # paths as strings
#         dataset_key = os.path.join(self.data_dir, self.dataset) + '/'
#         partition = [x.key for x in my_bucket.objects.filter(Prefix=dataset_key)]
#         print('Retrieving data from; ', dataset_key)
        
#         # get each file key
#         Y_files = [x for x in partition if x.find('/Y_train') != (-1)]
#         X_files_hyp = [x for x in partition if x.find('/X_train_hyp_') != (-1)]
#         X_files = X_files_hyp if self.w_hyp else set(partition) - set(Y_files) - set(X_files_hyp)
        
#         # sort (will be shuffled if shuffle=True)
#         X_files = sorted(X_files, key=lambda x: (len(x), x))
#         Y_files = sorted(Y_files, key=lambda x: (len(x), x))
        
#         return X_files, Y_files
    