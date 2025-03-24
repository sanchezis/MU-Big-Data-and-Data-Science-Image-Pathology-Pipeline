# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging

import sys
import glob
from pyspark.sql import SparkSession

from digital_pathology.spark import spark
from digital_pathology.process.tiatoolbox import NucleotidAndRegionExtractor

LOG_FILENAME = 'project.log'
APP_NAME = "Process several library features"

if __name__ == '__main__':
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    logging.info(sys.argv)

    sc = spark.sparkContext
    app_name = sc.appName
    logging.info("Application Initialized: " + app_name)
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        pass

    ##################################################
    NucleotidAndRegionExtractor().run(spark)
    ##################################################

    logging.info("Application Done: " + spark.sparkContext.appName)
    spark.stop()


# clear && poetry build && poetry run spark-submit  --master local --py-files dist/digital_pathology-*.whl  jobs/process_tiatoolbox.py
