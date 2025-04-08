import os
import warnings

import boto3
import pandas as pd
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

# ─── Suppress that Pydantic shadowing warning ─────────────────────────────────
warnings.filterwarnings(
    "ignore",
    message="Field name \"json\" in \"MonitoringDatasetFormat\" shadows an attribute",
)

# ─── Configuration ────────────────────────────────────────────────────────────
region     = os.environ.get("AWS_REGION", "us-east-1")
s3_bucket  = os.environ.get("S3_BUCKET", "fraud-detectml1")
fg_name    = os.environ.get("FEATURE_GROUP_NAME", "creditcard-fg1")

# ← Your CodeBuild service role ARN for SageMaker
role_arn   = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

# ─── Initialize SageMaker session ─────────────────────────────────────────────
session = Session(boto3.Session(region_name=region))

# ─── Load & prepare the top 5 rows ────────────────────────────────────────────
df = pd.read_csv("creditcard.csv").head(5)
df["event_time"] = pd.to_datetime("now")

# Reset index so we have a unique record identifier
df = df.reset_index().rename(columns={"index": "record_id"})

# ─── Build FeatureDefinition list ─────────────────────────────────────────────
feature_defs = []
for col, dtype in df.dtypes.items():
    if pd.api.types.is_integer_dtype(dtype):
        ftype = FeatureTypeEnum.INTEGRAL
    else:
        ftype = FeatureTypeEnum.FRACTIONAL

    feature_defs.append(
        FeatureDefinition(feature_name=col, feature_type=ftype)
    )

# ─── Create the FeatureGroup ─────────────────────────────────────────────────
fg = FeatureGroup(
    name=fg_name,
    sagemaker_session=session,
    feature_definitions=feature_defs,
)

fg.create(
    s3_uri=f"s3://{s3_bucket}/feature-store/",
    record_identifier_name="record_id",
    event_time_feature_name="event_time",
    role_arn=role_arn,
)

# ─── Ingest the DataFrame ─────────────────────────────────────────────────────
# Use data_frame= instead of records=
fg.ingest(
    data_frame=df,
    max_workers=3,
    wait=True
)

print(f"✅ Feature group '{fg_name}' created and ingested top 5 rows.")
