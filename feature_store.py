import warnings
import time
import uuid

import boto3
import pandas as pd
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup

# ─── Optional: suppress that pydantic warning ─────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="Field name \"json\".*shadows an attribute",
)

# ─── Configuration ────────────────────────────────────────────────────────────
REGION              = "us-east-1"
BUCKET_NAME         = "fraud-detectml1"
FEATURE_GROUP_NAME  = "creditcard-fg2"
S3_OFFLINE_STORE    = f"s3://{BUCKET_NAME}/feature-store/"
ROLE_ARN            = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

# ─── AWS Sessions ─────────────────────────────────────────────────────────────
boto3.setup_default_session(region_name=REGION)
boto_sess          = boto3.Session(region_name=REGION)
sagemaker_session  = sagemaker.Session(boto_session=boto_sess)
sm_client          = boto_sess.client("sagemaker", region_name=REGION)

# ─── Load & Clean Data ─────────────────────────────────────────────────────────
df = pd.read_csv("creditcard.csv").head(5).copy()

# Rename and ensure correct dtypes
df = df.rename(columns={"scaled_time": "Time", "scaled_amount": "Amount"})
df["record_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

# Drop any unsupported columns or nulls
# (we expect only floats and ints in V1–V28, Time, Amount, and Class)
df = df.dropna()

# ─── Build Feature Definitions ─────────────────────────────────────────────────
feature_defs = [
    FeatureDefinition("record_id", FeatureTypeEnum.STRING),
    FeatureDefinition("Time",      FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("Amount",    FeatureTypeEnum.FRACTIONAL),
    FeatureDefinition("Class",     FeatureTypeEnum.INTEGRAL),
] + [
    FeatureDefinition(f"V{i}", FeatureTypeEnum.FRACTIONAL) for i in range(1, 29)
]

# ─── Initialize FeatureGroup ───────────────────────────────────────────────────
fg = FeatureGroup(
    name=FEATURE_GROUP_NAME,
    feature_definitions=feature_defs,
    sagemaker_session=sagemaker_session
)

# ─── Delete existing FG if present ──────────────────────────────────────────────
try:
    sm_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
    print(f"⚠️  Feature group '{FEATURE_GROUP_NAME}' exists. Deleting...")
    sm_client.delete_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)

    # Poll until deletion completes
    print("⏳ Waiting for feature group to delete...")
    while True:
        try:
            sm_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
            time.sleep(5)
        except sm_client.exceptions.ResourceNotFound:
            print("✅ Previous feature group deleted.")
            break
except ClientError as e:
    if "ResourceNotFound" in str(e):
        print("✅ No existing feature group found; continuing.")
    else:
        raise

# ─── Create new FeatureGroup (offline only) ────────────────────────────────────
print(f"🚀 Creating feature group '{FEATURE_GROUP_NAME}'...")
fg.create(
    s3_uri=S3_OFFLINE_STORE,
    record_identifier_name="record_id",
    event_time_feature_name="Time",
    role_arn=ROLE_ARN,
    enable_online_store=False
)

# Poll until creation completes
print("⏳ Waiting for feature group to become ACTIVE...")
while True:
    resp = sm_client.describe_feature_group(FeatureGroupName=FEATURE_GROUP_NAME)
    status = resp["FeatureGroupStatus"]
    print(f"  • status: {status}")
    if status == "Created":
        print("✅ Feature group is ACTIVE.")
        break
    if status == "CreateFailed":
        raise RuntimeError("Feature group creation failed.")
    time.sleep(5)

# ─── Ingest Data ────────────────────────────────────────────────────────────────
print("📥 Ingesting data into feature store...")
fg.ingest(data_frame=df, max_workers=3, wait=True)
print(f"✅ Successfully ingested {len(df)} records into '{FEATURE_GROUP_NAME}'.")
