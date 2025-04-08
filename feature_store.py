import os
import warnings
import boto3
import pandas as pd
import time
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
from botocore.exceptions import ClientError

# Suppress warnings
warnings.filterwarnings("ignore", message="Field name \"json\".*")

# Config
region = os.environ.get("AWS_REGION", "us-east-1")
s3_bucket = "fraud-detectml1"
fg_name = "creditcard-fg2"
role_arn = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

# SageMaker session
boto_sess = boto3.Session(region_name=region)
session = Session(boto_session=boto_sess)
sm_client = boto_sess.client("sagemaker")

# Check if feature group exists, delete if yes
try:
    response = sm_client.describe_feature_group(FeatureGroupName=fg_name)
    print(f"⚠️ Feature Group '{fg_name}' already exists. Deleting...")
    sm_client.delete_feature_group(FeatureGroupName=fg_name)
    print("⏳ Waiting 30s for deletion to propagate...")
    time.sleep(30)
except ClientError as e:
    if "ResourceNotFound" in str(e):
        print(f"✅ Feature Group '{fg_name}' does not exist. Proceeding to create.")
    else:
        raise e

# Load and prepare data
df = pd.read_csv("creditcard.csv").head(5).copy()
df["event_time"] = pd.Timestamp.now().isoformat()
df = df.reset_index().rename(columns={"index": "record_id"})

# Clean column names
df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]

# Drop unsupported types
unsupported = df.select_dtypes(include=["object", "datetime64"]).columns.tolist()
if unsupported:
    print(f"⚠️ Dropping unsupported columns: {unsupported}")
    df = df.drop(columns=unsupported)

# Drop nulls
df = df.dropna()

# Define features
feature_defs = []
for col, dtype in df.dtypes.items():
    if col in ["record_id", "event_time"]:
        continue
    if pd.api.types.is_integer_dtype(dtype):
        ftype = FeatureTypeEnum.INTEGRAL
    else:
        ftype = FeatureTypeEnum.FRACTIONAL
    feature_defs.append(FeatureDefinition(feature_name=col, feature_type=ftype))

# Create feature group
fg = FeatureGroup(name=fg_name, sagemaker_session=session, feature_definitions=feature_defs)

fg.create(
    s3_uri=f"s3://{s3_bucket}/feature-store/",
    record_identifier_name="record_id",
    event_time_feature_name="event_time",
    role_arn=role_arn,
    description="Credit card fraud detection features"
)

# Ingest
fg.ingest(data_frame=df, max_workers=3, wait=True)

print(f"✅ FeatureGroup '{fg_name}' created and successfully ingested {len(df)} rows.")
