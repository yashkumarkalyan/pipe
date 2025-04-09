import os
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve

# ─── Configuration ────────────────────────────────────────────────────────────
region    = os.getenv("AWS_REGION", "us-east-1")
bucket    = os.getenv("S3_BUCKET", "fraud-detectml1")
role      = get_execution_role()  # assumes you're running in SageMaker or have STS permissions
session   = boto3.Session(region_name=region)
sm_session = sagemaker.Session(boto_session=session)

# ─── Retrieve the correct XGBoost container URI ───────────────────────────────
# This will pull from the official AWS ECR repo for XGBoost 1.5-1
image_uri = retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1",
    py_version="py3",
    instance_type="ml.m4.xlarge"
)

print(f"Using XGBoost image: {image_uri}")

# ─── Create the Estimator ─────────────────────────────────────────────────────
xgb = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path=f"s3://{bucket}/models/",
    sagemaker_session=sm_session
)

xgb.set_hyperparameters(
    objective="binary:logistic",
    num_round=100
)

# ─── Define inputs ────────────────────────────────────────────────────────────
train_input = TrainingInput(f"s3://{bucket}/data/train.csv", content_type="csv")
val_input   = TrainingInput(f"s3://{bucket}/data/test.csv",  content_type="csv")

# ─── Launch training ──────────────────────────────────────────────────────────
xgb.fit({"train": train_input, "validation": val_input})
print("✅ Training complete.")
