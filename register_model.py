import os
import boto3

region = os.environ['AWS_REGION']
s3_bucket = os.environ['S3_BUCKET']
mpg_name = os.environ['MODEL_PACKAGE_GROUP_NAME']

sm = boto3.client('sagemaker', region_name=region)

# create model package group if not exists
try:
    sm.create_model_package_group(
        ModelPackageGroupName=mpg_name,
        ModelPackageGroupDescription='CreditCard Fraud Detection MPG'
    )
    print("Created ModelPackageGroup.")
except sm.exceptions.ResourceInUse:
    print("ModelPackageGroup already exists.")

# register the latest model in S3
model_artifacts = f's3://{s3_bucket}/models/{os.environ.get("TRAIN_JOB_NAME")}/output/model.tar.gz'

response = sm.create_model_package(
    ModelPackageGroupName=mpg_name,
    ModelPackageDescription='XGBoost model for creditcard fraud',
    InferenceSpecification={
        'Containers': [{
            'Image': f'246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.5-1',
            'ModelDataUrl': model_artifacts
        }],
        'SupportedContentTypes': ['text/csv'],
        'SupportedResponseMIMETypes': ['text/csv']
    }
)
mp_arn = response['ModelPackageArn']
print(f"Registered model: {mp_arn}")
