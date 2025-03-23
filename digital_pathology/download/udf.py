# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging

import sys
import glob
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructType, ArrayType, StructField, TimestampType, LongType, BinaryType, IntegerType

import tifffile
from tiffslide import TiffSlide

from digital_pathology.TissueTileExtract import TissueTileExtractor

BINARY_FILES_SCHEMA = StructType(
    [
        StructField("path", StringType()),
        StructField("TimeStart", LongType()),
        StructField("TimeEnd", LongType()),
        StructField("modificationTime", TimestampType()),
        StructField("length", LongType()),
        StructField("bytes_string", StringType()),
        StructField("tiff_properties", StringType()),
        StructField("content", ArrayType(BinaryType())),
    ]
)



def pathology_image_quality_check(t):
    import numpy as np
    from digital_pathology.utils import norm_HnE
    
    bincount = np.bincount(t.flatten())
            
    try:
        norm, H, E = norm_HnE(t)
    except:
        pass
    
    if len(bincount)<=128:    
        return False
        
    if bincount.argmax() == 0 and \
        bincount.max() > bincount.mean()//2:
        return False
    
    try:
        if norm.mean() <= 40 or norm.mean() > 225 or \
            norm.std() < 35 :
            return False
    except:
        return False
    
    if bincount[:5].mean() > (1024*1024)//2 or \
        bincount[-10:].mean() > (1024*1024)//2:
        return False
    return True




@F.udf(returnType = BINARY_FILES_SCHEMA)
def download(bucket, key, out_path, DEBUG=True):
    # import io
    # import os
    import datetime
    import time

    start_ts = time.time_ns()
    # bucket = s3.Bucket(bucket) # 'camelyon-dataset'
    # object = bucket.Object(key) # CAMELYON17/images/patient_001_node_0.tif

    # buffer = None
    # image = None
    # slide = None

    # if not DEBUG:    
    #     file_stream = io.BytesIO()
    #     object.download_fileobj(file_stream)
    #     buffer = bytearray(file_stream.getvalue())

    # if buffer:
    #     image = tifffile.imread(buffer)
    #     slide = TiffSlide(buffer)

    # file_stream = io.BytesIO()
    # object.download_fileobj(file_stream)
    # #img = mplimg.imread(file_stream)
    
    # # print(file_stream.closed)
    # name = key.split('/')[-1]
    # # Write the stuff
    # with open(os.path.join(out_path, name), "wb") as f:
    #     f.write(file_stream.getbuffer())

    # tissue = TissueTileExtractor(f'../data/{name}', out_path, max_tiles=400)

    # tissue.extract_tiles( pathology_image_quality_check )
    
    # # # tifffile.imwrite(file_stream, [[0]])
    # # buffer = bytearray(file_stream.getvalue())
    # # return buffer

    return [ 
            None, 
            None,
            time.time_ns(),
            datetime.datetime.now(), 
            None, 
            None,
            None,
            [None,] 
            ] 

    return [ 
            os.path.join(bucket , key), 
            start_ts,
            time.time_ns(),
            datetime.datetime.now(), 
            len(buffer), 
            f"{len(buffer)//1024**2/1024} GB",
            slide.properties['tiff.ImageDescription'] if slide else None,
            [None,] 
            ] 
