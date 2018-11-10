import boto3


def clean_bucket_prefix(bucket_name, key_prefix):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for key in bucket.objects.all():
        if key.key.startswith(key_prefix):
            key.delete()
