import os
import time
import boto3
from botocore.exceptions import ClientError
from sagemaker.image_uris import retrieve

# ─── Config ───────────────────────────────────────────────────────────────────
region    = os.environ.get("AWS_REGION", "us-east-1")
bucket    = os.environ["S3_BUCKET"]
mpg_name  = os.environ["MODEL_PACKAGE_GROUP_NAME"]
train_job = os.environ.get("TRAIN_JOB_NAME")

sm = boto3.client("sagemaker", region_name=region)

# ─── Get training job ─────────────────────────────────────────────────────────
def get_training_job_name():
    global train_job
    if train_job:
        try:
            sm.describe_training_job(TrainingJobName=train_job)
            return train_job
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                print(f"⚠️  TRAIN_JOB_NAME '{train_job}' not found. Falling back...")
            else:
                raise

    print("🔍 Looking for latest completed training job...")
    jobs = sm.list_training_jobs(SortBy="CreationTime", SortOrder="Descending", MaxResults=10)
    for j in jobs["TrainingJobSummaries"]:
        if j["TrainingJobStatus"] == "Completed":
            train_job = j["TrainingJobName"]
            print(f"✅ Using training job: {train_job}")
            return train_job

    raise RuntimeError("❌ No completed training job found.")

# ─── Get model artifact path ──────────────────────────────────────────────────
def get_model_artifact_s3_path(job_name):
    desc = sm.describe_training_job(TrainingJobName=job_name)
    return desc["ModelArtifacts"]["S3ModelArtifacts"]

# ─── Delete all model packages in the group ───────────────────────────────────
def delete_model_packages_in_group(name):
    packages = sm.list_model_packages(ModelPackageGroupName=name)['ModelPackageSummaryList']
    for pkg in packages:
        pkg_name = pkg['ModelPackageArn']
        print(f"🗑️  Deleting model package: {pkg_name}")
        sm.delete_model_package(ModelPackageName=pkg_name)
        time.sleep(1)

# ─── Override MPG ─────────────────────────────────────────────────────────────
def override_model_package_group(name):
    try:
        sm.describe_model_package_group(ModelPackageGroupName=name)
        print(f"⚠️  ModelPackageGroup '{name}' exists. Deleting model packages...")
        delete_model_packages_in_group(name)
        print("🧹 All model packages deleted.")

        sm.delete_model_package_group(ModelPackageGroupName=name)
        print("🕐 Waiting for MPG to delete...")
        while True:
            try:
                sm.describe_model_package_group(ModelPackageGroupName=name)
                print("…still deleting MPG…")
                time.sleep(5)
            except ClientError as e:
                if e.response["Error"]["Code"] == "ValidationException":
                    print("✅ Deleted ModelPackageGroup.")
                    break
                else:
                    raise
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"ℹ️  No existing MPG '{name}', will create new.")
        else:
            raise

# ─── Main ─────────────────────────────────────────────────────────────────────
job_name = get_training_job_name()
model_artifacts = get_model_artifact_s3_path(job_name)

override_model_package_group(mpg_name)

sm.create_model_package_group(
    ModelPackageGroupName=mpg_name,
    ModelPackageGroupDescription="CreditCard Fraud Detection MPG"
)
print(f"✅ Created MPG '{mpg_name}'")

image_uri = retrieve("xgboost", region=region, version="1.5-1")
print(f"🧠 Using container: {image_uri}")

resp = sm.create_model_package(
    ModelPackageGroupName=mpg_name,
    ModelPackageDescription="XGBoost model for fraud detection",
    InferenceSpecification={
        "Containers": [{
            "Image": image_uri,
            "ModelDataUrl": model_artifacts
        }],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    },
    ModelApprovalStatus="Approved"
)
print(f"✅ Model registered: {resp['ModelPackageArn']}")
