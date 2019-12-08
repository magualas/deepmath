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
import progressbar


# config
BUCKET_NAME = 'sagemaker-cs281'
paths = {
    'train':'deephol-data-processed/proofs/human/train',
    'valid':'deephol-data-processed/proofs/human/valid',
    'test':'deephol-data-processed/proofs/human/test'
}

def run_extraction_pipeline(data_split=None):
    if not data_split:
        raise Exception('Need to specifiy if train, test or valid')
        
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
    maxval_bar = 400000
    pbar = progressbar.ProgressBar(maxval=maxval_bar)
    train_parsed = train_parsed  # remove 'take' part to iterate over the entire dataset
    for raw_record in pbar(train_parsed):
        fx, lx = raw_record[0], raw_record[1]
        features['goal'].append(fx['goal'])
        features['goal_asl'].append(fx['goal_asl'])
        features['thms'].append(fx['thms'])
        features['thms_hard_negatives'].append(fx['thms_hard_negatives'])
        labels['tac_id'].append(lx['tac_id'])
    print('Success: extracting data into arrays for goals, thms and tactics')

    # instantiate extractor object
    ex = extractor2.Extractor(params)

    # tokenize goals
    features['goal_ids'] = ex.tokenize(features['goal'], ex.vocab_table)
    print('Success: tokenizing goals')

    # tokenize hypotheses. this requires more work since there may be more than one hypothesis
    length = len(features['goal'])
    features['goal_asl_ids'] =  [[]] * length
    print('1')
    pbar2 = progressbar.ProgressBar()
    print('2')
    for i in pbar2(range(length)):
        # pad all hypotheses to be of length 1000
        temp = ex.tokenize(features['goal_asl'][i], ex.vocab_table)
        hypo_list = [[]] * len(temp)
        for j in range(len(temp)):
            l = len(temp[j])
            h = tf.pad(temp[j], [[0, 1000-l]], constant_values=0)
            hypo_list[j] = h
        features['goal_asl_ids'][i] = np.array(hypo_list)
    print('Success: tokenizing hypotheses')

    # free up memory
    del features['goal']
    del features['goal_asl']
    del features['thms']
    del features['thms_hard_negatives']

    # make into an array and upload to s3
    # features: goals
    goals_array = np.array(features['goal_ids'])
    upload_np_to_s3(goals_array, os.path.join(paths[data_split], 'goal_ids.csv'))
    del goals_array
    del features['goal_ids']
    print('Uploaded goals successfully')
    
    # features: hypotheses
    pbar3 = progressbar.ProgressBar(maxval=maxval_bar)
    for i, hyp in pbar3(enumerate(features['goal_asl_ids'])):
        upload_np_to_s3(np.array(hyp), 
                        os.path.join(paths[data_split], 'goal_asl_ids_{}.csv'.format(i)))
    del features['goal_asl_ids']
    print('Uploaded hypotheses successfully')
    
    # labels: tactid ids
    labels_array = np.array(labels['tac_id'])
    upload_np_to_s3(labels_array, os.path.join(paths[data_split], 'tac_id.csv'))
    del labels_array
    print('Uploaded labels (tactics) successfully')
    

def upload_np_to_s3(array, object_name):    
    # save localy
    local_filename = '/tmp/temp.csv'
    np.savetxt(local_filename, array, delimiter=',')
    
    # s3 upload
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(local_filename, BUCKET_NAME, object_name)