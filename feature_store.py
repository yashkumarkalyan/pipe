import os
import boto3
import pandas as pd
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

# config
region       = os.environ['AWS_REGION']
s3_bucket    = os.environ['S3_BUCKET']
fg_name      = os.environ['FEATURE_GROUP_NAME']
role_arn     = boto3.client('sts', region_name=region).get_caller_identity()['Arn']

session = Session(boto3.Session(region_name=region))

# load top 5 rows
df = pd.read_csv('creditcard.csv').head(5)
df['event_time'] = pd.to_datetime('now')
df = df.reset_index().rename(columns={'index':'record_id'})

# build FeatureDefinition list
feature_defs = []
for col, dtype in df.dtypes.items():
    # choose integral vs fractional
    if dtype in ['int64', 'int32']:
        ftype = FeatureTypeEnum.INTEGRAL
    else:
        ftype = FeatureTypeEnum.FRACTIONAL
    feature_defs.append(FeatureDefinition(feature_name=col, feature_type=ftype))

# create the FeatureGroup
fg = FeatureGroup(name=fg_name,
                  sagemaker_session=session,
                  feature_definitions=feature_defs)

fg.create(
    s3_uri=f's3://{s3_bucket}/feature-store/',
    record_identifier_name='record_id',
    event_time_feature_name='event_time',
    role_arn=role_arn
)

# ingest the records
records = df.to_dict(orient='records')
fg.ingest(records=records, max_workers=3, wait=True)

print(f"Feature group '{fg_name}' created and ingested top 5 rows.")
