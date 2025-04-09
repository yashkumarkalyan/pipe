import os
import time
import boto3

# Load environment variables
region = os.environ['AWS_REGION']
s3_bucket = os.environ['S3_BUCKET']
mpg_name = os.environ['MODEL_PACKAGE_GROUP_NAME']
role_arn = os.environ['SAGEMAKER_ROLE_ARN']  # You MUST export this before running

sm = boto3.client('sagemaker', region_name=region)

# 1. Get latest approved model package
models = sm.list_model_packages(
    ModelPackageGroupName=mpg_name,
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)

if not models['ModelPackageSummaryList']:
    raise ValueError("No models found in the model package group.")

model_package_arn = models['ModelPackageSummaryList'][0]['ModelPackageArn']

# 2. Create model from model package
model_name = "batch-inference-model-" + str(int(time.time()))
print(f"Creating model: {model_name}")

sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role_arn,
    PrimaryContainer={
        'ModelPackageName': model_package_arn
    }
)

# 3. Create batch transform job
job_name = "batch-transform-" + str(int(time.time()))
print(f"Starting batch transform job: {job_name}")

transform_input = {
    'DataSource': {
        'S3DataSource': {
            'S3DataType': 'S3Prefix',
            'S3Uri': f's3://{s3_bucket}/data/test.csv'
        }
    },
    'ContentType': 'text/csv',
    'SplitType': 'Line'
}

output_config = {
    'S3OutputPath': f's3://{s3_bucket}/batch-output/'
}

transform_resources = {
    'InstanceType': 'ml.m4.xlarge',  # Free tier-compatible
    'InstanceCount': 1
}

sm.create_transform_job(
    TransformJobName=job_name,
    ModelName=model_name,
    TransformInput=transform_input,
    TransformOutput=output_config,
    TransformResources=transform_resources
)

print(f"Transform job {job_name} submitted. Waiting 150 seconds before checking status...")
time.sleep(150)

# 4. Check job status
status = sm.describe_transform_job(TransformJobName=job_name)['TransformJobStatus']
print(f"Batch job status: {status}")
