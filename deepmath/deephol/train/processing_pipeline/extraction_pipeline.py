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
PARTITION_SIZE = 5000
data_paths = {
    'train':'deephol-data-processed/proofs/human/train',
    'valid':'deephol-data-processed/proofs/human/valid',
    'test':'deephol-data-processed/proofs/human/test'
}

def run_extraction_pipeline(data_split=None):
    """
    """
    if not data_split:
        raise Exception('Need to specifiy if train, test or valid')

    # make tf dataset of parsed examples
    params = ingestor.get_params()
    train_data = data.get_train_dataset(params)
    parser = data.tristan_parser
    train_parsed = train_data.map(functools.partial(parser, params=params))

    # set features and labels
    features = {'goal': [], 'goal_asl': [], 'thms': [], 'thms_hard_negatives': []}
    labels = {'tac_id': []}

    # iterate over dataset to extract data into arrays
    train_parsed = train_parsed # CHANGE HERE
    bar1 = progressbar.ProgressBar()
    for raw_record in bar1(train_parsed):
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

    # features['goal_ids'] is now an array of size N x 1000
    features['goal_ids'] = features['goal_ids'].numpy()
    print('Number of training examples:', len(features['goal_ids']))
    print('Size of training examples:', len(features['goal_ids'][0]))

    # features['goal_asl_ids'] is now an array of size  N x ? x 1000
    length = len(features['goal_asl_ids'])
    for i in range(length):
        features['goal_asl_ids'][i] = [hypothesis.numpy() 
                                       for hypothesis in features['goal_asl_ids'][i]]
    print('Number of training examples:', len(features['goal_asl_ids']))
    print('Number of hypotheses for an example:', len(features['goal_asl_ids'][0]))

    # features['tactic_ids'] is now an array of size N x 1
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
            hypotheses[i] = np.concatenate(hypotheses[i])  # concatenate hypotheses in a given hypothesis list
            hypotheses[i] = hypotheses[i][hypotheses[i] != 0]  # remove zeroes in between
            hypotheses[i] = hypotheses[i][0:3000]  # truncate to max hyp length = 3000 chars (< than 10% of data
            len_conc = len(hypotheses[i]) # pad with zeroes to make length 3000 (to save as csv)
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
        train_example = np.concatenate((X_train[i], hypotheses[i]))  # concatenate goal and hypotheses
        train_example = train_example[train_example != 0]  # remove zeroes in between
        train_example = train_example[0:3000]  # truncate to max hyp length of 3000 chars (less than 10% of data
        len_conc = len(train_example)  # pad with zeroes to make length 3000 (to save as csv)
        train_example = np.pad(train_example, (0, 3000-len_conc), mode='constant')
        X_train_hyp.append(np.asarray(train_example, dtype='float64').tolist())
    X_train_hyp = np.array(X_train_hyp)
    print(np.shape(X_train_hyp))    
    
    # save to s3
    partition_size = PARTITION_SIZE if len(Y_train) > PARTITION_SIZE else len(Y_train) 
    n_partitions = len(Y_train) // partition_size
    print(len(Y_train), partition_size, n_partitions)
    for i, split in enumerate(np.array_split(X_train, n_partitions), 1):
        upload_np_to_s3(split, os.path.join(data_paths[data_split], 'X_train_{}.csv'.format(i)))
    print('Uploaded all X_train files')
    for i, split in enumerate(np.array_split(X_train_hyp, n_partitions), 1):
        upload_np_to_s3(split, os.path.join(data_paths[data_split], 'X_train_hyp_{}.csv'.format(i)))
    print('Uploaded all X_train_hyp files')
    upload_np_to_s3(Y_train, os.path.join(data_paths[data_split], 'Y_train.csv'))
    print('Uploaded Y_train file')
    
    return 'Success'

def get_partition_and_labels(w_hyp=True, data_set='train'):
    """ Create a dictionary called partition where:
        - in partition['train']: a list of training IDs
        - in partition['validation']: a list of validation IDs
    """
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(BUCKET_NAME)
    partition = {'train': [x.key for x in my_bucket.objects.filter(Prefix=paths['train'])], 
                 'test': [x.key for x in my_bucket.objects.filter(Prefix=paths['test'])],
                 'valid':[x.key for x in my_bucket.objects.filter(Prefix=paths['valid'])]}
    
    y_file = [x for x in partition[data_set] if x.find('Y_train') != -1]
    if w_hyp:
        X_files = [x for x in partition[data_set] if x.find('X_train_hyp') != (-1)]
    else:
        X_files = [x for x in partition[data_set] if x.find('X_train_hyp') == (-1) and x.find('Y_train') == -1]
    
    return X_files, y_file


def upload_np_to_s3(array, object_name):    
    # save localy
    local_filename = '/tmp/temp.csv'
    np.savetxt(local_filename, array, delimiter=',')
    
    # s3 upload
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(local_filename, BUCKET_NAME, object_name)
    
    
def split_array_batches(array):
    """ split in equal length arrays """
    n_examples = len(array)
    partition_size = 4096
    end_to_skip = n_examples % partition_size  # have partitions of equal length
    print('skipping {} examples to have equal length batches'.format(end_to_skip))
    n_partitions = (n_examples - end_to_skip) / partition_size 

    print('partition size: {}'.format(partition_size))
    print('shape of truncated array: ', array[:-end_to_skip].shape)

    splits = np.split(array[:-end_to_skip], n_partitions)
    
    return splits