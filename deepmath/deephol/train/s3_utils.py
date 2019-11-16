"""

s3_utils.py

"""


import os

def s3_connect():
    # s3 configuration
    config = {
        'AWS_REGION':'us-east-2',                    # Region for the S3 bucket, this is not always needed. Default is us-east-1.
        'S3_ENDPOINT':'s3.us-east-2.amazonaws.com',  # The S3 API Endpoint to connect to. This is specified in a HOST:PORT format.
        'S3_USE_HTTPS':'1',                        # Whether or not to use HTTPS. Disable with 0.
        'S3_VERIFY_SSL':'1',  
    }

    os.environ.update(config)