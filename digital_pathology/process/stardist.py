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
        return

    def run(self, spark):
        @udf(BooleanType())
        def extract_nucleotid(col):
            import skimage
            from stardist.models import StarDist2D
            from stardist.data import test_image_nuclei_2d
            from stardist.plot import render_label
            from csbdeep.utils import normalize

            try:
                # Load model once per executor process
                if not hasattr(extract_nucleotid, 'model_he'):
                    extract_nucleotid.model_he = StarDist2D.from_pretrained('2D_versatile_he')
                model_he = extract_nucleotid.model_he
            except:
                return False
            
            image_file_name = col
            
            try:
                image = skimage.io.imread(image_file_name)
                labels, _ = model_he.predict_instances(normalize(image))
                skimage.io.imsave(f'{image_file_name}_nucleotid_labels.png', labels)
            except:
                return False
            
            return True

        # Check installation across cluster
        downloaded = spark.read.parquet(os.path.join('data', '1-download.parquet'))

        downloaded.show(truncate=False)

        downloaded = downloaded\
                        .withColumn(
                            "nucleotid_filename", 
                            extract_nucleotid(F.concat( F.lit(
                                os.path.join(
                                    'data',
                                    'patient_extracts', 
                                    )),
                                F.lit(os.path.sep),
                                F.col('filename') ))
                        )

        downloaded.write.mode('overwrite').parquet(os.path.join('data', '2-stardist-nucleotids.parquet'))

