"""

extraction_pipeline.py

by Jeff, Manu and Tanc

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


BUCKET_NAME = 'sagemaker-cs281'
config = {
    'AWS_REGION':'us-east-2',        
    'S3_ENDPOINT':'s3.us-east-2.amazonaws.com', 
    'S3_USE_HTTPS':'1',                 
    'S3_VERIFY_SSL':'1',  
}
os.environ.update(config)


class My_DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_set='train',batch_size=64,
                 w_hyp=False, n_channels=1, n_classes=41, shuffle=True):
        self.dim = 3000 if w_hyp else 1000 
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.w_hyp = w_hyp
        self.data_set = data_set
        
        # paths
        X_paths, Y_path = self.get_partition_and_labels()
        self.features_keys_lst = X_paths
        self.label_key = Y_path[0]
#         self.n = self._get_n_rows_parition(self.features_keys_lst[0])
        
        # readers
        self._initialize_readers()        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, id_lst=[]):
        'Generate one batch of data'
        X, y = next(self.reader_X), next(self.reader_Y)

        return X.values, y.values

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _get_n_rows_parition(self, key):
        s3_client = boto3.client('s3') 
        req = s3_client.select_object_content(
             Bucket=BUCKET_NAME,
             Key=key,
             ExpressionType='SQL',
             Expression='SELECT COUNT(*) FROM s3object',
             InputSerialization = {'CSV': {'FileHeaderInfo': 'Use'}},
             OutputSerialization = {'CSV': {}},
        )
        mess = next(req['Payload']._event_generator)
        print(mess.payload)
        print(mess)
        
        return int(mess.payload)
    
    def _initialize_readers(self):
        # must change to all features
#         obj_X = s3_client.get_object(Bucket=BUCKET_NAME, Key=self.features_keys_lst[0]) 
        path_X = os.path.join('s3://', BUCKET_NAME, self.features_keys_lst[0])
        self.reader_X = pd.read_csv(path_X, chunksize=self.batch_size, 
                                    header=None, engine='python')
        
        path_Y = os.path.join('s3://', BUCKET_NAME, self.label_key)
#         obj_Y = s3_client.get_object(Bucket=BUCKET_NAME, Key=self.label_key)
        self.reader_Y = pd.read_csv(path_Y, chunksize=self.batch_size,
                                    header=None, engine='python')
        
        return None
    
    def get_partition_and_labels(self):
        """ Create a dictionary called partition where:
            - in partition['train']: a list of training IDs
            - in partition['validation']: a list of validation IDs
        """
        w_hyp=self.w_hyp
        s3_r = boto3.resource('s3')
        my_bucket = s3_r.Bucket(BUCKET_NAME)
        data_paths = {'train':'deephol-data-processed/proofs/human/train',
                    'valid':'deephol-data-processed/proofs/human/valid',
                    'test':'deephol-data-processed/proofs/human/test'}
        partition = {'train': [x.key for x in my_bucket.objects.filter(Prefix=data_paths['train'])], 
                     'test': [x.key for x in my_bucket.objects.filter(Prefix=data_paths['test'])],
                     'valid':[x.key for x in my_bucket.objects.filter(Prefix=data_paths['valid'])]}

        print('Retrieving data from {}'.format(data_paths[self.data_set]))
        y_file = [x for x in partition[self.data_set] if x.find('Y_train') != -1]
        if w_hyp:
            X_files = [x for x in partition[self.data_set] if x.find('X_train_hyp') != (-1)]
        else:
            X_files = [x for x in partition[self.data_set] if x.find('X_train_hyp') == (-1) and x.find('Y_train') == -1]

        return sorted(X_files, key=lambda x: len(x)), y_file
