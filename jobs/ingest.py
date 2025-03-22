import logging

import sys
from pyspark.sql import SparkSession

from digital_pathology.spark import spark
from digital_pathology.ingest.aws import AWS_ingestion

LOG_FILENAME = 'project.log'
APP_NAME = "WordCount"

if __name__ == '__main__':
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    logging.info(sys.argv)

    if len(sys.argv) is not 2:
        logging.warning("Output path is needed.")
        sys.exit(1)

    sc = spark.sparkContext
    app_name = sc.appName
    logging.info("Application Initialized: " + app_name)
    output_path = sys.argv[1]
    
    AWS_ingestion('camelyon-dataset').run(spark, output_path)
    
    logging.info("Application Done: " + spark.sparkContext.appName)
    spark.stop()


# clear && poetry run spark-submit  --master local  --py-files dist/digital_pathology-*.whl   jobs/ingest.py  out/0-extract.parquet
