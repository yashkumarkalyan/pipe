import os
import boto3
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

region = os.environ['AWS_REGION']
s3_bucket = os.environ['S3_BUCKET']

role = get_execution_role()
session = boto3.Session(region_name=region)

# XGBoost container
container = f'246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.5-1'

# estimator
xgb = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{s3_bucket}/models/'
)

xgb.set_hyperparameters(
    objective='binary:logistic',
    num_round=100
)

train_input = TrainingInput(f's3://{s3_bucket}/data/train.csv', content_type='csv')
test_input  = TrainingInput(f's3://{s3_bucket}/data/test.csv',  content_type='csv')

xgb.fit({'train': train_input, 'validation': test_input})
print("Training complete.")
