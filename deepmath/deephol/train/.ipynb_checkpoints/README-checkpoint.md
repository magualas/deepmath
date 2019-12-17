# CS 281 final project

By Jeff, Tanc and Manu


## Structure of code repo
This README is located within the /deepmath/deephol/train/ folder of the HOList github repository which we forked
From the original README: This directory contains training code used for the WaveNet experiments described in "HOList: An Environment for Machine Learning of Higher-Order Theorem Proving" (https://arxiv.org/abs/1904.03241). This code is provided for reference, but due to dependency issues it builds but does not run. 

We then have the following packages to process and train models for tactic selections:

* /packages
    * generator.py: data generator to train on keras model.fit_generator()
    * utils.py: utilities function, AWS and tensorflow config
    * sagemaker_deploy.py: script to run python-sagemaker-sdk
    
* /processing_pipeline
    * extraction_pipeline_np.ipynb: notebook to process all data
    * extraction_pipeline.py: script version of the notebook
    * ingestor.py
    * extractor2.py
    
* /Dockerfile
    * train_and_deploy.py: python script that defines the behaviours of our model to create a sagemaker train instance

