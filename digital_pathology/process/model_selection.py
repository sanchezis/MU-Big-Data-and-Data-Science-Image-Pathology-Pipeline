# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import logging


class ResNetModel(object):
    
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        return 
    
    def run(self, spark):
        
        from tiatoolbox.utils.misc import download_data, imread
        from tiatoolbox.utils.visualization import overlay_prediction_mask
        from tiatoolbox.wsicore.wsireader import WSIReader

        from urllib import request
        import certifi
        import ssl

        
        logging.warning('********************** RUN **********************')
        
        context = ssl.create_default_context(cafile=certifi.where())
        https_handler = request.HTTPSHandler(context=context)
        opener = request.build_opener(https_handler)
        request.install_opener(opener)

        model_file_name = "tissue_mask_model.pth"
        download_data(
            "https://tiatoolbox.dcs.warwick.ac.uk//models/seg/fcn-tissue_mask.pth",
            model_file_name,
        )
        
        return
    