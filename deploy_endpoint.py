import os
import time
import boto3
from sagemaker import get_execution_role

region = os.environ['AWS_REGION']
mpg_name = os.environ['MODEL_PACKAGE_GROUP_NAME']
endpoint_name = os.environ['ENDPOINT_NAME']

sm = boto3.client('sagemaker', region_name=region)

# get latest model package
models = sm.list_model_packages(
    ModelPackageGroupName=mpg_name,
    SortBy='CreationTime',
    SortOrder='Descending',
    MaxResults=1
)
mp_arn = models['ModelPackageSummaryList'][0]['ModelPackageArn']

# create model
model_name = 'creditcard-model-' + str(int(time.time()))
role = get_execution_role()

sm.create_model(
    ModelName=model_name,
    PrimaryContainer={'ModelPackageName': mp_arn},
    ExecutionRoleArn=role
)

# endpoint config
config_name = model_name + '-config'
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InstanceType': 'ml.t2.medium',  # Free-tier eligible instance
        'InitialInstanceCount': 1
    }]
)

# create/update endpoint
existing = [e['EndpointName'] for e in sm.list_endpoints()['Endpoints']]
if endpoint_name in existing:
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"✅ Updated my endpoint: {endpoint_name}")
else:
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"✅ Created endpoint: {endpoint_name}")
