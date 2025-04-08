import os
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

# config
s3_bucket = os.environ['S3_BUCKET']
region = os.environ['AWS_REGION']
input_csv = 'creditcard.csv'
train_key = 'data/train.csv'
test_key  = 'data/test.csv'

# load locally (or from S3)
df = pd.read_csv(input_csv)
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_df = pd.concat([y_train, X_train], axis=1)
test_df = pd.concat([y_test, X_test], axis=1)

# save to local
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

# upload to S3
s3 = boto3.client('s3', region_name=region)
s3.upload_file('train.csv', s3_bucket, train_key)
s3.upload_file('test.csv',  s3_bucket, test_key)

print("Preprocessing done. Uploaded train/test to S3.")
