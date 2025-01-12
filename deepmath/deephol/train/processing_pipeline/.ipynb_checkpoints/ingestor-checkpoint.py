"""
ingestor.py

Who is the owner of this file?

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
print("Tesor Flow Version:", tf.__version__, "Ingest File")

import boto3
# from sagemaker import get_execution_role
tf.compat.v1.enable_eager_execution()

import utils
import data
import extractor
import s3_utils

TRAIN = tf.estimator.ModeKeys.TRAIN
EVAL = tf.estimator.ModeKeys.EVAL
PREDICT = tf.estimator.ModeKeys.PREDICT
SOURCE_DATASETDIR = 0
SOURCE_LOOPDIR = 1
WAIT_SECONDS = 60


# s3 config
s3_utils.s3_connect()
# role = get_execution_role()
bucket='sagemaker-cs281'
data_key = 'deephol-data/deepmath/deephol/proofs/human'
ddir = 's3://{}/{}'.format(bucket, data_key)
evalddir = None


# hyperparameters
class DataInfo(object):
    def __init__(self,dataset_dir,eval_dataset_dir):
        self.dataset_dir = dataset_dir
        self.eval_dataset_dir = eval_dataset_dir
        self.goal_vocab = 'vocab_goal_ls.txt'
        self.thm_vocab = 'vocab_thms_ls.txt'
        self.vocab = 'vocab_ls.txt'
        self.truncate_size = 1000
        self.ratio_neg_examples=7
        self.ratio_max_hard_negative_examples=5
        self.batch_size = 4
        
        
    def generate(self):
        return {'dataset_dir': self.dataset_dir, 
                'eval_dataset_dir': self.eval_dataset_dir,
                'goal_vocab': self.goal_vocab,
                'thm_vocab': self.thm_vocab,
                'vocab': self.vocab,
                'truncate_size': self.truncate_size,
                'ratio_neg_examples': self.ratio_neg_examples, 
                'ratio_max_hard_negative_examples': self.ratio_max_hard_negative_examples, 
                'batch_size': self.batch_size} 

def get_params():
    
    d = DataInfo(ddir,evalddir)
    hparams = d.generate()
    
    return utils.Params(**hparams)    
    
def get_input_fn(is_train):
    """ 
    get input functions, call to get features and labels
    
    WTF does train parsed. 
    """
    d = DataInfo(ddir,evalddir)
    hparams = d.generate()
    params = utils.Params(**hparams)

    if is_train:
         input_fn = data.get_input_fn(dataset_fn=data.get_train_dataset, mode=TRAIN, params=params, shuffle_queue=10000, repeat=False)
    
    else:
         input_fn = data.get_input_fn(dataset_fn=data.get_eval_dataset, mode=EVAL, params=params, shuffle_queue=10000, repeat=False)
    
    return input_fn, params
    
    
    