import os
import time
import boto3

region = os.environ['AWS_REGION']
s3_bucket = os.environ['S3_BUCKET']
mpg_name = os.environ['MODEL_PACKAGE_GROUP_NAME']

sm = boto3.client('sagemaker', region_name=region)

# get latest approved model package
models = sm.list_model_packages(
    ModelPackageGroupName=mpg_name,
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)
mp_arn = models['ModelPackageSummaryList'][0]['ModelPackageArn']

# create batch transform job
job_name = 'batch-transform-'+str(int(time.time()))
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
output_config = {'S3OutputPath': f's3://{s3_bucket}/batch-output/'}

sm.create_transform_job(
    TransformJobName=job_name,
    ModelName=mp_arn,
    TransformInput=transform_input,
    TransformOutput=output_config,
    TransformResources={'InstanceType': 'ml.m5.large', 'InstanceCount': 1}
)

print(f"Started Batch Transform job {job_name}, sleeping for 150s...")
time.sleep(150)

# optional: check status
status = sm.describe_transform_job(TransformJobName=job_name)['TransformJobStatus']
print(f"Batch job status: {status}")
