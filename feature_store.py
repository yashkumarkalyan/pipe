import boto3
import pandas as pd
import sagemaker
import uuid
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker import get_execution_role
from botocore.exceptions import ClientError
import time

# Set region and session
REGION = "us-east-1"
boto3.setup_default_session(region_name=REGION)
boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

# Define constants
FEATURE_GROUP_NAME = "creditcard-fg2"
BUCKET_NAME = "fraud-detectml1"
S3_PATH = f"s3://{BUCKET_NAME}/feature-store/"

# Load and preprocess the dataset
df = pd.read_csv("creditcard.csv")
df = df.rename(columns={"scaled_time": "Time", "scaled_amount": "Amount"})
df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

# Initialize SageMaker FeatureGroup
feature_group = FeatureGroup(name=FEATURE_GROUP_NAME, sagemaker_session=sagemaker_session)

# Delete existing feature group if it exists
sm_client = boto_session.client("sagemaker", region_name=REGION)
try:
    sm_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
    print(f"Feature group '{FEATURE_GROUP_NAME}' exists. Deleting...")
    sm_client.delete_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
    waiter = sm_client.get_waiter("feature_group_deleted")
    waiter.wait(FeatureGroupName=FEATURE_GROUP_NAME)
    print("Previous feature group deleted.")
except ClientError as e:
    if "ResourceNotFound" in str(e):
        print("No existing feature group found. Proceeding...")
    else:
        raise e

# Define feature definitions
feature_defs = [
    FeatureDefinition(feature_name="record_id", feature_type=FeatureTypeEnum.STRING),
    FeatureDefinition(feature_name="Time", feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="Amount", feature_type=FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition(feature_name="Class", feature_type=FeatureTypeEnum.INTEGRAL),
] + [
    FeatureDefinition(feature_name=f"V{i}", feature_type=FeatureTypeEnum.FRACTIONAL) for i in range(1, 29)
]

# Create the feature group
try:
    feature_group.create(
        s3_uri=S3_PATH,
        record_identifier_name="record_id",
        event_time_feature_name="Time",
        role_arn=role,
        enable_online_store=False,
        feature_definitions=feature_defs
    )
    print(f"Feature group '{FEATURE_GROUP_NAME}' created.")
except Exception as e:
    print(f"Failed to create feature group: {e}")
    raise

# Wait until feature group is ready
try:
    feature_group.wait_for_create()
    print("Feature group is active.")
except Exception as e:
    print(f"Error waiting for feature group creation: {e}")
    raise

# Ingest records
try:
    print("Ingesting top 5 rows to feature store...")
    feature_group.ingest(data_frame=df.head(5), max_workers=3, wait=True)
    print("Data ingested successfully.")
except Exception as e:
    print(f"Error during data ingestion: {e}")
    raise
