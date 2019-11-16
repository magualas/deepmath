# python file that is used to ingest data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
print(tf.__version__)

import boto3
from sagemaker import get_execution_role
tf.compat.v1.enable_eager_execution()

import utils
import data
import extractor

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT

SOURCE_DATASETDIR = 0
SOURCE_LOOPDIR = 1

WAIT_SECONDS = 60

# s3 configuration

config = {
#     'AWS_ACCESS_KEY_ID':'AKIAR66VYUC6IKHLEWOV',            # Credentials only needed if connecting to a private endpoint
#     'AWS_SECRET_ACCESS_KEY':'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',
    'AWS_REGION':'us-east-2',                    # Region for the S3 bucket, this is not always needed. Default is us-east-1.
    'S3_ENDPOINT':'s3.us-east-2.amazonaws.com',  # The S3 API Endpoint to connect to. This is specified in a HOST:PORT format.
    'S3_USE_HTTPS':'1',                        # Whether or not to use HTTPS. Disable with 0.
    'S3_VERIFY_SSL':'1',  
}

os.environ.update(config)

role = get_execution_role()
bucket='sagemaker-cs281'
data_key = 'deephol-data/deepmath/deephol/proofs/human'

ddir = 's3://{}/{}'.format(bucket, data_key)
evalddir = None

# hyperparameters

class DataInfo(object):

    def __init__(self,dataset_dir,eval_dataset_dir):
        self.dataset_dir = dataset_dir
        self.eval_dataset_dir = eval_dataset_dir
        self.ratio_neg_examples=7
        self.ratio_max_hard_negative_examples=5
        self.batch_size = 4
        
    def generate(self):
        return {'dataset_dir': self.dataset_dir, 'eval_dataset_dir': self.eval_dataset_dir, 'ratio_neg_examples': 
                self.ratio_neg_examples, 'ratio_max_hard_negative_examples': self.ratio_max_hard_negative_examples, 
                'batch_size': self.batch_size,
               } 

d = DataInfo(ddir,evalddir)
hparams = d.generate()

params = utils.Params(**hparams)

# obtain raw training dataset
train_data = data.get_holparam_dataset(TRAIN, params)
eval_data = data.get_holparam_dataset(EVAL, params)

#obtain parsed training dataset
train_parsed = train_data.map(functools.partial(data.pairwise_thm_parser, params=params))
input_fn = data.get_input_fn(dataset_fn=data.get_train_dataset, mode=TRAIN, params=params, shuffle_queue=10000, repeat=False)