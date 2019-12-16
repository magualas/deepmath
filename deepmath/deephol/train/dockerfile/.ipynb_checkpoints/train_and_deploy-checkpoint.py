"""

training_and_deploy.py 

code for amazon sagemaker training

"""

import os
import numpy as np


import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput
import sagemaker
from sagemaker import get_execution_role

import generator


# config
INPUT_TENSOR_NAME = "inputs_input"
SIGNATURE_NAME = "serving_default"
W_HYP = False
LEARNING_RATE = 0.001
BATCH_SIZE = 64
VOCAB_SIZE = 1254
INPUT_LENGTH = 3000 if W_HYP else 1000
EMBEDDING_DIM = 128
# sagemaker_session = sagemaker.Session()
# role = get_execution_role()s


# WHEN do we find training_dir?
# training_dir = 'deephol-data-processed/proofs/human'


def keras_model_fn(hyperparameters):
    # architecture
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=INPUT_LENGTH, name='inputs'))
    model.add(SpatialDropout1D(0.2))
    model.add(CuDNNLSTM(4))
    model.add(Dense(41, activation='softmax'))
    
    # optimizer: can use keras class to add parameters
    opt = 'adam'
    
    # loss: can use keras class to add parameters
    loss = 'categorical_crossentropy'

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    
    return model


def serving_input_fn(hyperparameters):
    # Here it concerns the inference case where we just need a placeholder to store
    # the incoming images ...
    tensor = tf.placeholder(tf.float64, shape=[None, INPUT_LENGTH])
    inputs = {INPUT_TENSOR_NAME: tensor}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def train_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE, data_dir=training_dir)


def eval_input_fn(training_dir, hyperparameters):
    return _input(tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE, data_dir=training_dir)


def _input(mode, batch_size, data_dir):
    assert os.path.exists(data_dir), ("Unable to find input directory, check path")

    if mode == tf.estimator.ModeKeys.TRAIN:
        datagen = generator.Keras_DataGenerator(data_dir=data_dir, dataset='train', 
                                                w_hyp=W_HYP, batch_size=batch_size)
    elif mode == tf.estimator.ModeKeys.VALID:
        datagen = generator.Keras_DataGenerator(data_dir=data_dir, dataset='valid',
                                                w_hyp=W_HYP, batch_size=batch_size)
    elif mode == tf.estimator.ModeKeys.TEST:
        datagen = generator.Keras_DataGenerator(data_dir=data_dir, dataset='test',
                                                w_hyp=W_HYP, batch_size=batch_size)
    else:
        raise Exception('Invalid tf estimator mode key')

    # get batch
    features, labels = next(datagen)

    return {INPUT_TENSOR_NAME: features}, labels


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    args, _ = parser.parse_known_args()
    # another function that does the real work 
    # (and make the code cleaner)
    run_training(args)