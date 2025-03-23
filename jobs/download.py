import logging

import sys
import glob
from pyspark.sql import SparkSession

from digital_pathology.spark import spark
from digital_pathology.download.files import DownloadPathologyData

LOG_FILENAME = 'project.log'
APP_NAME = "Download and Extract Characteristics"

if __name__ == '__main__':
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    logging.info(sys.argv)

    if len(sys.argv) != 3:
        logging.warning("Output path is needed.")
        sys.exit(1)

    sc = spark.sparkContext
    app_name = sc.appName
    logging.info("Application Initialized: " + app_name)
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    DownloadPathologyData('camelyon-dataset').run(spark, input_path, output_path)

    logging.info("Application Done: " + spark.sparkContext.appName)
    spark.stop()


# clear && poetry run spark-submit  --master local  --py-files dist/digital_pathology-*.whl   jobs/download.py 'data/0-extract.parquet'  data/patient_extracts
