# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging

import os
import sys
import glob
from pyspark.sql import SparkSession

from digital_pathology.spark import spark
from digital_pathology.process.model_selection import ResNetModel

LOG_FILENAME = 'project.log'
APP_NAME = "Process several library features"

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    logging.info(sys.argv)

    if len(sys.argv) != 1:
        logging.warning("No need for config.")
        sys.exit(1)

    # # Initialize Spark with OpenSlide config
    # spark = SparkSession.builder \
    #     .appName("TIAToolboxWSI") \
    #     .config("spark.executorEnv.OPENSLIDE_PATH", "/usr/lib/openslide") \
    #     .config("spark.executorEnv.OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES") \
    #     .getOrCreate()
    
    sc = spark.sparkContext
    app_name = sc.appName
    
    logging.info("Application Initialized: " + app_name)
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        pass

    ##################################################
    logging.warning('*******************************************')
    
    from pyspark.sql.functions import udf
    from openslide import OpenSlide
    from pyspark.sql.types import BooleanType
    
    @udf(BooleanType())
    def verify_openslide():
        try:
            from openslide import OpenSlide
            return True
        except:
            return False

    # Check installation across cluster
    spark.range(10).withColumn("openslide_installed", verify_openslide()).show()    
    
    ResNetModel().run(spark)
    logging.warning('*******************************************')
    ##################################################

    logging.info("Application Done: " + spark.sparkContext.appName)
    spark.stop()




















# clear && poetry build && poetry run spark-submit  --master local  --py-files dist/digital_pathology-*.whl  --files lib/*  jobs/process_tiatoolbox_model_sel.py
