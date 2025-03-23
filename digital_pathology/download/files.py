# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging

import os
import sys
import glob
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from digital_pathology.spark import spark
from digital_pathology.TissueTileExtract import TissueTileExtractor
from digital_pathology.utils import download_image
from digital_pathology.download.udf import pathology_image_quality_check

# from digital_pathology.download.udf import download
# from digital_pathology.utils import image_from_s3

"""_summary_

SOLAMENTE SE DESCARGAN 5 pacientes, dadas las limitaciones 
de tamaño que poseen las imágenes.

"""

class DownloadPathologyData(object):

    def __init__(self, Bucket, credentials=None):
        self.Bucket = Bucket # 'camelyon-dataset'
        pass

    def run(self, spark: SparkSession, input_path: str, output_path: str):
        ingestion = spark.read.parquet(input_path)

        ingestion = ingestion.withColumn('image',   F.regexp_extract ('filename',  '.*/(\w+\..*)', 1) ) 
        ingestion.show(truncate=False)

        transform = ingestion\
            .filter( # (F.col('loc').like('image%')) &
                    (F.col('project').like('CAMELYON17'))
                    )\
            .withColumn('out_dir', F.concat( F.lit('data'), F.lit('/'), F.col('image') )  ) \
            .filter(    ( F.col('image').rlike(r'patient_0[0|1]')  ) |
                        ( F.col('loc')=='') |
                        ( F.col('type').isin( ['py','csv', 'txt'] ))
                    )\
            .filter(  (F.col('image')!='') ) \
            .sort('image')

        transform_1 = transform\
                        .filter(  (F.col('type')=='tif')  &  (~F.col('image').rlike(r'(_mask)')) )

        
        logging.info('***************************************************************')
        logging.info(f'{input_path}  ->   {output_path}')
        logging.info('***************************************************************')

        logging.info('TAKING ONLY 10 random Patients')
        output = transform_1.sample(.20).limit(10)
        list_of_files = output.select('filename', 'out_dir').collect()

        logging.info(f'{list_of_files}')

        for i, e in enumerate(list_of_files):
            logging.info(f'Downloading: {i+1} {e}')

            # S3 configuration
            bucket_name = 'camelyon-dataset'
            s3_path = e['filename']

            download_image(bucket_name, s3_path, e['out_dir'])        

            tissue = TissueTileExtractor(e['out_dir'], output_path, max_tiles=1000)
            tissue.extract_tiles( pathology_image_quality_check )

            file_path = e['out_dir']

            try:
                os.remove(file_path)
                logging.warning(f"{file_path} deleted successfully")
            except FileNotFoundError:
                logging.warning(f"The file {file_path} does not exist")
            except PermissionError:
                logging.warning(f"Permission denied to delete {file_path}")
            except Exception as e:
                logging.warning(f"Error deleting file: {str(e)}")

        result = spark.createDataFrame ( 
                        pd.DataFrame ( glob.glob("../data/patient_extracts/*.tif"),  columns=['downloaded_tiles'] ) , 
                        schema=T.StructType([ T.StructField("downloaded_tiles", T.StringType()), ])
                    )

        result.write.mode('overwrite').parquet('data/1-download.parquet')

        logging.info('***************************************************************')

