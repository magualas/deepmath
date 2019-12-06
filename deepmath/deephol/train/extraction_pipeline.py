"""

extraction_pipeline.py

by Jeff, Manu and Tanc

"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import boto3
# print(tf.__version__)

import ingestor
import extractor2
import data
import functools


# config
BUCKET_NAME = 'sagemaker-cs281'
paths = {
    'train':'deephol-data-processed/proofs/human/train',
    'valid':'',
    'test':''
}


def run_extraction_pipeline():
    # get dataset parameters
    params = ingestor.get_params()

    # make tf dataset of parsed examples
    train_data = data.get_train_dataset(params)
    parser = data.tristan_parser
    train_parsed = train_data.map(functools.partial(parser, params=params))

    # set features and labels
    features = {'goal': [], 'goal_asl': [], 'thms': [], 'thms_hard_negatives': []}
    labels = {'tac_id': []}

    # iterate over dataset to extract data into arrays. 
    # remove 'take' part to iterate over the entire dataset
    train_parsed = train_parsed.take(2000)
    for raw_record in train_parsed:
        fx, lx = raw_record[0], raw_record[1]
        features['goal'].append(fx['goal'])
        features['goal_asl'].append(fx['goal_asl'])
        features['thms'].append(fx['thms'])
        features['thms_hard_negatives'].append(fx['thms_hard_negatives'])
        labels['tac_id'].append(lx['tac_id'])
    print('Done extracting data into arrays for goals, thms and tactics')

    # instantiate extractor object
    ex = extractor2.Extractor(params)

    # tokenize goals
    temp = ex.tokenize(features['goal'], ex.vocab_table)
    
    # pad all goals to be of length 1000
    goal_list = []
    for j in range(len(temp)):
            l = len(temp[j])
            h = tf.pad(temp[j], [[0, 1000-l]], constant_values=0)
            goal_list.append(h)
    features['goal_ids'] = goal_list
    print('Done tokenizing goals')

    # tokenize hypotheses. this requires more work since there may be more than one hypothesis
    length = len(features['goal'])
    features['goal_asl_ids'] = []
    for i in range(length):
        temp = ex.tokenize(features['goal_asl'][i], ex.vocab_table)
        
        # pad all hypotheses to be of length 1000
        hypo_list = []
        for j in range(len(temp)):
            l = len(temp[j])
            h = tf.pad(temp[j], [[0, 1000-l]], constant_values=0)
            hypo_list.append(h)
        features['goal_asl_ids'].append(hypo_list)
    print('Done tokenizing hypotheses')

    # free up memory
    del features['goal']
    del features['goal_asl']
    del features['thms']
    del features['thms_hard_negatives']

    # make into an array and upload to s3
    # features: goals
    goals_array = np.array(features['goal_ids'])
    upload_np_to_s3(goals_array, os.path.join(paths['train'], 'goal_ids.csv'))
    del goals_array
    
    # features: hypotheses
    for i, hyp in enumerate(features['goal_asl_ids']):
        upload_np_to_s3(np.array(hyp), 
                        os.path.join(paths['train'], 'goal_asl_ids_{}.csv'.format(i)))
    del features['goal_asl_ids']
    
    # labels: tactid ids
    labels_array = np.array(labels['tac_id'])
    upload_np_to_s3(labels_array, os.path.join(paths['train'], 'tac_id.csv'))
    del labels_array
    
    print('Successfully uploaded to s3 the goals, hypotheses and labels')
    

def upload_np_to_s3(array, object_name):    
    # save localy
    local_filename = '/tmp/temp.csv'
    np.savetxt(local_filename, array, delimiter=',')
    
    # s3 upload
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(local_filename, BUCKET_NAME, object_name)