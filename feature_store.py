import os
import boto3
import pandas as pd
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

# config
region = os.environ['AWS_REGION']
s3_bucket = os.environ['S3_BUCKET']
fg_name = os.environ['FEATURE_GROUP_NAME']

session = Session(boto3.Session(region_name=region))
feature_store_client = boto3.client('sagemaker-featurestore-runtime', region_name=region)

# load top 5 rows
df = pd.read_csv('creditcard.csv').head(5)
df['event_time'] = pd.to_datetime('now')

# define feature definitions
feature_defs = []
for col, dtype in zip(df.columns, df.dtypes):
    dtype_str = 'Integral' if dtype in ['int64'] else 'Fractional'
    feature_defs.append({'FeatureName': col, 'FeatureType': dtype_str})

# create FeatureGroup
fg = FeatureGroup(name=fg_name,
                  sagemaker_session=session,
                  feature_definitions=feature_defs)

fg.create(
    s3_uri=f's3://{s3_bucket}/feature-store/',
    record_identifier_name='index',
    event_time_feature_name='event_time',
    role_arn=session.get_caller_identity_arn()
)

# ingest records
records = df.reset_index().to_dict(orient='records')
fg.ingest(records=records, max_workers=3, wait=True)

print(f"Feature group {fg_name} created and ingested top 5 rows.")
