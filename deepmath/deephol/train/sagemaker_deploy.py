"""

sagemaker_deploy.py


This repository contains example notebooks that show how to apply machine learning and deep learning in Amazon SageMaker
* https://github.com/awslabs/amazon-sagemaker-examples

Within, some examples of training keras model on sagemaker
* https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_keras

"""

import os
import sagemaker
import numpy as np
from sagemaker.tensorflow import TensorFlow
from tensorflow.python.keras.preprocessing.image import load_img


# IAM role
ON_SAGEMAKER_NOTEBOOK = False
sagemaker_session = sagemaker.Session()
if ON_SAGEMAKER_NOTEBOOK:
    role = sagemaker.get_execution_role()
else:
    role = 'arn:aws:iam::135202906300:user/jeff_admin'

# config
bucket = 'sagemaker-cs281'
key = 'deephol-data-processed/proofs/human'   # Path from the bucket's root to the dataset
train_instance_type='ml.p3.2xlarge'     # The type of EC2 instance which will be used for training
deploy_instance_type='ml.p3.2xlarge'    # The type of EC2 instance which will be used for deployment
n_GPUs = 1
hyperparameters={
    "learning_rate": 1e-4,
    "decay": 1e-6
}
train_input_path = "s3://{}/{}/train/".format(bucket, key)
validation_input_path = "s3://{}/{}/validation/".format(bucket, key)


# estimator
estimator = TensorFlow(
  entry_point=os.path.join(os.path.dirname(__file__), "train_and_deploy.py"),     # Your entry script
  role=role,
  framework_version="1.12.0",               # TensorFlow's version
  hyperparameters=hyperparameters,
  training_steps=1000,
  evaluation_steps=100,
  train_instance_count=n_GPUs,                   # "The number of GPUs instances to use"
  train_instance_type=train_instance_type,
)

print("Training ...")
estimator.fit({'training': train_input_path, 'eval': validation_input_path})
