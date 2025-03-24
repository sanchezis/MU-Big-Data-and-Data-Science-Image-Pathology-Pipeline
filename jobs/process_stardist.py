# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging

import sys
import glob

from digital_pathology.spark import spark
from digital_pathology.process.stardist import NucleotidExtractor

LOG_FILENAME = 'project.log'
APP_NAME = "Process several library features"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv)

    # if len(sys.argv) != 3:
    #     logging.warning("Output path is needed.")
    #     sys.exit(1)

    sc = spark.sparkContext
    app_name = sc.appName
    logging.info("Application Initialized: " + app_name)
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    except:
        pass

    ##################################################
    from stardist.models import StarDist2D

    from pyspark.sql.functions import udf
    from pyspark.sql.types import BooleanType
    
    @udf(BooleanType())
    def verify_StarDist2D():
        try:
            from stardist.models import StarDist2D
            return True
        except:
            return False

    # Check installation across cluster
    spark.range(10).withColumn("StarDist2D_installed", verify_StarDist2D()).show()    
    NucleotidExtractor().run(spark)
    ##################################################

    logging.info("Application Done: " + spark.sparkContext.appName)
    spark.stop()


# clear && poetry build && poetry run spark-submit  --master local --py-files dist/digital_pathology-*.whl  jobs/process_stardist.py
