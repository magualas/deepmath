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
PARTITION_SIZE = 50000
paths = {
    'train':'deephol-data-processed/proofs/human/train',
    'valid':'deephol-data-processed/proofs/human/valid',
    'test':'deephol-data-processed/proofs/human/test'
}

def run_extraction_pipeline(data_split=None):
    """
    """
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

    # iterate over dataset to extract data into arrays
    train_parsed = train_parsed.take(100) # CHANGE HERE
    for raw_record in train_parsed:
        fx, lx = raw_record[0], raw_record[1]
        features['goal'].append(fx['goal'])
        features['goal_asl'].append(fx['goal_asl'])
        features['thms'].append(fx['thms'])
        features['thms_hard_negatives'].append(fx['thms_hard_negatives'])
        labels['tac_id'].append(lx['tac_id'])

    # instantiate extractor object
    ex = extractor2.Extractor(params)

    # tokenize goals
    features['goal_ids'] = ex.tokenize(features['goal'], ex.vocab_table)

    # tokenize hypotheses
    length = len(features['goal'])
    features['goal_asl_ids'] = []
    for i in range(length):
        temp = ex.tokenize(features['goal_asl'][i], ex.vocab_table)
        features['goal_asl_ids'].append(temp)

    # free memory
    del features['goal']
    del features['goal_asl']
    del features['thms']
    del features['thms_hard_negatives']

    # features['goal_ids'] is now an array of size 2000 x 1000
    features['goal_ids'] = features['goal_ids'].numpy()
    print('Number of training examples:', len(features['goal_ids']))
    print('Size of training examples:', len(features['goal_ids'][0]))

    # features['goal_asl_ids'] is now an array of size 2000 x ? x 1000
    length = len(features['goal_asl_ids'])
    for i in range(length):
        features['goal_asl_ids'][i] = [hypothesis.numpy() 
                                       for hypothesis in features['goal_asl_ids'][i]]
    print('Number of training examples:', len(features['goal_asl_ids']))
    print('Number of hypotheses for an example:', len(features['goal_asl_ids'][0]))
    print('Size of each hypothesis:', len(features['goal_asl_ids'][0][0]))

    # features['tactic_ids'] is now an array of size 2000 x 1
    labels['tac_id'] = [i.numpy() for i in labels['tac_id']]
    print('Number of training examples:', len(labels['tac_id']))

    # convert goals to numpy arrays
    goals = np.array(features['goal_ids'])
    print(np.shape(goals))

    # convert goal hypotheses to numpy arrays and concatenate
    hypotheses = features['goal_asl_ids']
    length_hyp = len(hypotheses)
    for i in range(length_hyp):
        if (len(hypotheses[i]) != 0):
            # concatenate hypotheses in a given hypothesis list
            hypotheses[i] = np.concatenate(hypotheses[i])
            # remove zeroes in between
            hypotheses[i] = hypotheses[i][hypotheses[i] != 0]
            # truncate to max hypothesis length of 3000 characters, i.e. truncating less than 10% of data
            hypotheses[i] = hypotheses[i][0:3000]
            # pad with zeroes to make length 3000 (to save as csv)
            len_conc = len(hypotheses[i])
            hypotheses[i] = np.pad(hypotheses[i], (0, 3000-len_conc), mode='constant')
        else:
            hypotheses[i] = np.zeros(3000, dtype = 'int32')

    np.set_printoptions(threshold=np.sys.maxsize)
    print(np.shape(hypotheses))

    # convert tactics to numpy arrays and one-hot encode
    a = np.array(labels['tac_id'])
    tactics = np.zeros((a.size, 40+1))
    tactics[np.arange(a.size),a] = 1
    print(np.shape(tactics))

    X_train, Y_train = goals, tactics
    print(np.shape(X_train))
    print(np.shape(Y_train))
    print(np.shape(hypotheses))

    # create feature matrix with goals and hypotheses
    length = len(X_train)
    X_train_hyp = []
    for i in range(length):
        # concatenate goal and hypotheses
        train_example = np.concatenate((X_train[i], hypotheses[i]))
        # remove zeroes in between
        train_example = train_example[train_example != 0]
        # truncate to max hypothesis length of 3000 characters, i.e. truncating less than 10% of data
        train_example = train_example[0:3000]
        # pad with zeroes to make length 3000 (to save as csv)
        len_conc = len(train_example)
        train_example = np.pad(train_example, (0, 3000-len_conc), mode='constant')
        X_train_hyp.append(np.asarray(train_example, dtype='float64').tolist())
    X_train_hyp = np.array(X_train_hyp)
    print(np.shape(X_train_hyp))    
    
    # save to s3
    partition_size = PARTITION_SIZE if len(Y_train) > 50000 else len(Y_train) 
    n_partitions = len(Y_train) // partition_size
    print(len(Y_train), partition_size, n_partitions)
    for i, split in enumerate(np.array_split(X_train, n_partitions)):
        upload_np_to_s3(split, os.path.join(paths[data_split], 'X_train_{}.csv'.format(i)))
    print('Uploaded all X_train files')
    for i, split in enumerate(np.array_split(X_train_hyp, n_partitions)):
        upload_np_to_s3(split, os.path.join(paths[data_split], 'X_train_hyp_{}.csv'.format(i)))
    print('Uploaded all X_train_hyp files')
    upload_np_to_s3(Y_train, os.path.join(paths[data_split], 'Y_train.csv'))
    print('Uploaded all Y_train file')
    
    return X_train, X_train_hyp, Y_train


def upload_np_to_s3(array, object_name):    
    # save localy
    local_filename = '/tmp/temp.csv'
    np.savetxt(local_filename, array, delimiter=',')
    
    # s3 upload
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(local_filename, BUCKET_NAME, object_name)