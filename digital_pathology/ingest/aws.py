# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

from digital_pathology.spark import spark

import boto3 

import pandas as pd

from botocore.exceptions import ClientError
from botocore.handlers import disable_signing

import pyspark.sql.functions as F
from pyspark.sql.types import StructField, StructType, StringType
from pyspark.sql import SparkSession


class AWS_ingestion(object):

    def __init__(self, Bucket, credentials=None):
        self.Bucket = Bucket # 'camelyon-dataset'
        pass

    def get_s3_data(self):
        resource = boto3.resource('s3')
        resource.meta.client.meta.events.register('choose-signer.s3.*', disable_signing)

        my_bucket = resource.Bucket(self.Bucket)
        di = []

        for my_bucket_object in my_bucket.objects.all():
            di.append(( my_bucket_object.key, ) )
        
        return di

    def run(self, spark: SparkSession, output_path: str = 'out/0-extract.parquet'):
        schema = StructType(
            [
                StructField("filename", StringType()),
            ]
        )

        ingestion = spark.createDataFrame( self.get_s3_data(), schema=schema)

        ingestion = ingestion.withColumn('type',      F.regexp_extract ('filename',  '\.(.+)$', 1) )
        ingestion = ingestion.withColumn('loc',       F.regexp_extract ('filename',  '/(.+)/.*', 1) )
        ingestion = ingestion.withColumn('project',   F.regexp_extract ('filename',  '(\w+)/.*', 1) )
        
        ingestion.show(truncate=False)

        ingestion.write.mode('overwrite').parquet(output_path)

