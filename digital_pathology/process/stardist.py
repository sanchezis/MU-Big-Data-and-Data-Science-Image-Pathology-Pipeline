# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging
import os

import pandas as pd
import glob
import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType


class NucleotidExtractor(object):
    def __init__(self):
        from stardist.models import StarDist2D
        self.model_he = StarDist2D.from_pretrained('2D_versatile_he')
        return

    def run(self, spark):
        @udf(BooleanType())
        def verify_StarDist2D():
            from stardist.models import StarDist2D
            
            return True

        # Check installation across cluster
        spark.range(10).withColumn("StarDist2D_installed", verify_StarDist2D()).show()    





        downloaded = spark.read.parquet(os.path.join('data', '1-download.parquet'))

        downloaded.show(truncate=False)
