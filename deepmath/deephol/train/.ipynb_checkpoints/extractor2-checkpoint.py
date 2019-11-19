"""Extractor for HOLparam models. Tokenizes goals and theorems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
print("Tesor Flow Version:", tf.__version__, "Extactor 2")

import utils

class Extractor(object):
  """Extract terms/thms and tokenize based on vocab.

  Attributes:
    params: Hyperparameters.
    goal_table: Lookup table for goal vocab embeddings.
    thms_table: Lookup table for theorem parameter vocab embeddings.
    add_negative: Integer multiple ratio of negative examples to positives.
    all_thms: List of all training thms as strings.
    random_negatives: A batch of negative thm examples.
    goal_closed_negative_iterator: A iterator for negative goal closed examples.
  """

  def __init__(self, params):
    """Inits Extractor class with hyperparameters."""
    self.params = params

    # Create vocab lookup tables from existing vocab id lists.
    dataset_dir = params['dataset_dir']
    goal_file = os.path.join(dataset_dir, params['goal_vocab'])
    self.goal_table = utils.vocab_table_from_file(goal_file)
    if params['thm_vocab'] is not None:
      thms_file = os.path.join(dataset_dir, params['thm_vocab'])
      self.thms_table = utils.vocab_table_from_file(thms_file)
    else:
      self.thms_table = self.goal_table

  def tokenize(self, tm, table):
    """Tokenizes tensor string according to lookup table."""
    tm = tf.strings.join(['<START> ', tf.strings.strip(tm), ' <END>'])
    # Remove parentheses - they can be recovered for S-expressions.
    tm = tf.strings.regex_replace(tm, r'\(', ' ')
    tm = tf.strings.regex_replace(tm, r'\)', ' ')
    words = tf.strings.split(tm)
    # Truncate long terms.
    words = tf.sparse.slice(words, [0, 0],
                            [tf.shape(words)[0], self.params.truncate_size])

    word_values = words.values
    id_values = tf.to_int32(table.lookup(word_values))
    ids = tf.SparseTensor(words.indices, id_values, words.dense_shape)
    ids = tf.sparse_tensor_to_dense(ids)
    return ids

  def extractor(self, features, labels):
    """Converts 'goal' features and 'thms' labels to list of ids by vocab."""

    if 'goal' not in features:
      raise ValueError('goal feature missing.')
    if 'tac_id' not in labels:
      raise ValueError('tac_id label missing.')

    # tokenize 'goal' and 'thms'.
    tf.add_to_collection('goal_string', features['goal'])
    features['goal_ids'] = self.tokenize(features['goal'], self.goal_table)
    del features['goal']
    if 'thms' in features:
      tf.add_to_collection('thm_string', features['thms'])
      features['thm_ids'] = self.tokenize(features['thms'], self.thms_table)
      del features['thms']
      del features['thms_hard_negatives']

    return features, labels