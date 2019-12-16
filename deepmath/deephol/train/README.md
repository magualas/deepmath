# CS 281 final project

By Jeff, Tanc and Manu


## Structure of code repo
* train_and_deploy.py
    * python script that defines the behaviours of our model by the
* sagemaker_deploy.py
    * script to run python-sagemaker-sdk
* generator.py
    * data generator to train on keras model.fit_generator()
* extraction_pipeline_np.ipynb
    * notebook to process all data
    * must adapt data.py > get_train_dataset() to 'train'/'valid'/'test'


## Old
* This directory contains training code used for the WaveNet experiments described in "HOList: An Environment for Machine Learning of Higher-Order Theorem Proving" (https://arxiv.org/abs/1904.03241).
* This code is provided for reference, but due to dependency issues it builds but does not run.
