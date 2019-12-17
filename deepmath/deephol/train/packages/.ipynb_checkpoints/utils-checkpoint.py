'''
Utils
'''
<<<<<<< HEAD
import boto3
import os

import json

=======

import boto3
import os
import json
>>>>>>> 9afc01edfcbf33f6925277f1e3bc1e5bf18041c2
import numpy as np

from keras import backend as K


def aws_setup():
    BUCKET_NAME = 'sagemaker-cs281'
    config = {
    'AWS_REGION':'us-east-2',                    
    'S3_ENDPOINT':'s3.us-east-2.amazonaws.com',  
    'S3_USE_HTTPS':'1',                       
    'S3_VERIFY_SSL':'1',  
    }
    s3_client = boto3.client('s3') 
    os.environ.update(config)
    print("AWS SETUP SHOULD BE COMPLETE, we are on " + str(s3_client))
<<<<<<< HEAD
    
=======
>>>>>>> 9afc01edfcbf33f6925277f1e3bc1e5bf18041c2


def GPU_checker():
    n_GPUs = len(K.tensorflow_backend._get_available_gpus())
    print("You are runnning an instance with " + str(n_GPUs) + " GPU's")

<<<<<<< HEAD
=======
    
>>>>>>> 9afc01edfcbf33f6925277f1e3bc1e5bf18041c2
def GPU_count():
    n_GPUs = len(K.tensorflow_backend._get_available_gpus())
    return n_GPUs


<<<<<<< HEAD

=======
>>>>>>> 9afc01edfcbf33f6925277f1e3bc1e5bf18041c2
def history_saver_bad(history, LOSS_FILE_NAME):
    '''this needs to be improved'''
    loss_history = history.history['loss']    
    
    #TODO: read file if it exists merge two files rather than overwriting 
    numpy_loss_history = np.array(loss_history)
    np.savetxt("training_logs/"+ LOSS_FILE_NAME +".csv", numpy_loss_history, delimiter=",")
    
    # save full history json
    with open('training_logs/'+ LOSS_FILE_NAME +'.json', 'w') as f:
        history_dict = vars(history)
        try:
            del history_dict['model']
        except:
            print('no model in vars dict')
        json.dump(history_dict, f)
        
    print("SAVED SOME LOGS -- OVERWROTE OLD LOGS -- SOMEONE NEEDS TO FIX THIS")
    
<<<<<<< HEAD
=======
def save_history_obj_as_json(history, filename):
    pass
    
>>>>>>> 9afc01edfcbf33f6925277f1e3bc1e5bf18041c2
    
    
    
    
    