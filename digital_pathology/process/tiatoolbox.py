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

from tiatoolbox import logger
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

try:
    import pyspark
    from pyspark.sql.functions import col, isnan, when, count,to_date,year,month,expr,hour,dayofweek,lower,array_remove,collect_list,lit
    from pyspark.sql.functions import pandas_udf,split
    from pyspark.sql.types import ArrayType, DoubleType, StringType
    from pyspark.sql.types import StructField,StructType,StringType,DoubleType,FloatType,IntegerType
    import pyspark.sql.functions as F
except:
    pass


# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from tiatoolbox import logger
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

import numpy as np
import histomicstk as htk
import skimage
import scipy as sp




BINARY_FILES_SCHEMA = StructType(
    [
        # StructField("tile_prediction",  BinaryType()),
        StructField("bins", StringType()),
        StructField("out", StringType()),
        # StructField("content", ArrayType(BinaryType())),
    ]
)


class NucleotidAndRegionExtractor(object):
    def __init__(self):

        return

    def run(self, spark):

        @F.udf(returnType = BINARY_FILES_SCHEMA)
        def extract_tumor(path, img_name, img_path, DEBUG=True):
            import logging
            import os
            import numpy as np
            from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor


            # Disable logging to avoid issues
            import logging
            logger = logging.getLogger()
            logger.setLevel(logging.CRITICAL)
            
            label_names_dict = {
                0: "Tumour",
                1: "Stroma",
                2: "Inflamatory",
                3: "Necrosis",
                4: "Others",
            }

            # Tile prediction
            bcc_segmentor = SemanticSegmentor(
                pretrained_model=model_file_name, # "fcn_resnet50_unet-bcss", # Ensure this path is worker-accessible
                num_loader_workers=0,    # Avoid Multiprocessing in UDF CRUCIAL: Disable multiprocessing
                batch_size=4,
            )
            
            out_location = os.path.join(img_path, f"sample_tile_results/{img_name}")
            logging.warning(out_location)
            
            import shutil
            #os.rmdir(f"sample_tile_results/{img[2]}")
            try:
                shutil.rmtree(out_location, exist_ok=True)
            except:
                pass
            
            output = bcc_segmentor.predict(
                [path],
                save_dir=out_location,
                mode="tile",
                resolution=1.0,
                units="baseline",
                patch_input_shape=[1024, 1024],
                patch_output_shape=[512, 512],
                stride_shape=[512, 512],
                on_gpu=False,
                crash_on_exception=True,
            )
            
            tile_prediction_raw = np.load(
                output[0][1] + f"{img_name}.raw.0.npy",
            ) 
            
            tile_prediction = np.argmax(
                tile_prediction_raw,
                axis=-1,
            ) 
            bins = np.bincount(tile_prediction.flatten())
            out = str( list( zip (label_names_dict.values(),  np.round( bins / np.sum(bins) * 100, 4)  ) ) )
            tile = imread(path)

            return [ 
                    # Image.fromarray(tile_prediction), 
                    str(bins),
                    out,
                    # [tile,] 
                    ] 


        @udf(BooleanType())
        def extract_nucleotid(image_file_name):
            # Hyperparameters

            # segment foreground
            foreground_threshold = 60

            # detect and segment nuclei using local maximum clustering
            local_max_search_radius = 10

            # run adaptive multi-scale LoG filter
            min_radius = 10
            max_radius = 15

            # filter out small objects
            min_nucleus_area = 80

            stainColorMap = {
                'hematoxylin': [0.65, 0.70, 0.29],
                'eosin':       [0.07, 0.99, 0.11],
                'dab':         [0.27, 0.57, 0.78],
                'null':        [0.0, 0.0, 0.0]
            }

            # specify stains of input image
            stain_1 = 'hematoxylin'   # nuclei stain
            stain_2 = 'eosin'         # cytoplasm stain
            stain_3 = 'null'          # set to null of input contains only two stains

            # create stain matrix
            W = np.array([  stainColorMap[stain_1],
                            stainColorMap[stain_2],
                            stainColorMap[stain_3]]).T


            image = skimage.io.imread( image_file_name )
            im_input = image[:, :, :3]
            
            # Reinhard parameters for LAB color space
            target_mu = [8.74108109, -0.12440419, 0.0444982]
            target_sigma = [0.6135447, 0.10987245, 0.2859532]

            # perform reinhard color normalization
            im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, target_mu, target_sigma)

            # perform standard color deconvolution
            im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains

            # get nuclei/hematoxylin channel
            im_nuclei_stain = im_stains[:, :, 0]

            im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(im_nuclei_stain < foreground_threshold)

            im_log_max, im_sigma_max = htk.filters.shape.cdog(
                im_nuclei_stain, im_fgnd_mask,
                sigma_min=min_radius * np.sqrt(2),
                sigma_max=max_radius * np.sqrt(2)
            )

            im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(im_log_max, im_fgnd_mask, local_max_search_radius)

            im_nuclei_seg_mask = htk.segmentation.label.area_open(im_nuclei_seg_mask, min_nucleus_area).astype(int)

            # compute nuclei properties
            objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

            return len(objProps)

        
        model_file_name = os.path.join('data', 'tissue_mask_model.pth')

        spark.sparkContext.addFile(model_file_name)
        spark.sparkContext.addFile(model_file_name[:-4]+'.lock')

        images_path = os.path.join('data', 'patient_extracts')


        # Check installation across cluster
        downloaded = spark.read.parquet(os.path.join('data', '1-download.parquet'))

        downloaded.show(truncate=False)

        downloaded = downloaded\
                        .repartition(2)\
                        .withColumn(
                            "found_nucleotids", 
                            extract_nucleotid(
                                F.concat( F.lit(
                                os.path.join(
                                    'data',
                                    'patient_extracts', 
                                    )),
                                F.lit(os.path.sep),
                                F.col('filename') )
                            )
                        )\
                        .withColumn('image',   
                                extract_tumor( 
                                            F.lit(images_path),
                                            F.col('filename') ,
                                            F.lit(images_path)
                                            ) 
                        ) 

        downloaded.write.mode('overwrite').parquet(os.path.join('data', '3-nucleotids.parquet'))

