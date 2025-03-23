# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import os
import logging
import numpy as np
import large_image
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.annotations_and_masks.annotation_and_mask_utils import (
    get_image_from_htk_response
)
from skimage.filters import threshold_otsu
from PIL import Image

class TissueTileExtractor:
    """Extracts tissue tiles from whole slide images"""

    def __init__(self, slide_path, output_dir, tile_size=1024, tissue_threshold=95.0, max_tiles=10_000):
        self.slide_path = slide_path
        self.name = self.slide_path.split(os.sep)[-1]
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.tissue_threshold = tissue_threshold
        self.max_tiles = max_tiles
        self.ts = large_image.getTileSource(slide_path)
        self.mask, self.scale_factor = self._create_tissue_mask()
        logging.warning(self.name)

    def _create_tissue_mask(self):
        """Create tissue mask using Reinhard normalization + Otsu thresholding"""
        # Get thumbnail for mask generation
        thumbnail = self.ts.getRegion(
            format=large_image.tilesource.TILE_FORMAT_NUMPY,
            output=dict(maxWidth=1024)
        )[0]
        
        # Reinhard parameters for LAB color space
        target_mu = [8.74108109, -0.12440419, 0.0444982]
        target_sigma = [0.6135447, 0.10987245, 0.2859532]

        # Apply color normalization
        normalized = reinhard(
            thumbnail[:, :, :3],
            target_mu=target_mu,
            target_sigma=target_sigma
        )
        
        # Convert to grayscale
        gray_img = np.dot(normalized, [0.2989, 0.5870, 0.1140])
        
        # Calculate Otsu's threshold using scikit-image
        thresh = threshold_otsu(gray_img)
        mask = (gray_img > thresh).astype(np.uint8) * 255
        
        # Calculate scale factor between mask and original image
        metadata = self.ts.getMetadata()
        scale_factor = metadata['sizeX'] / mask.shape[1]
        
        return mask, scale_factor

    def _get_tissue_percentage(self, x, y):
        """Calculate tissue percentage for a given tile position"""
        mask_x = int(x / self.scale_factor)
        mask_y = int(y / self.scale_factor)
        mask_size = int(self.tile_size / self.scale_factor)
        
        mask_region = self.mask[
            mask_y:mask_y+mask_size,
            mask_x:mask_x+mask_size
        ]
        return np.mean(mask_region) * 100

    def extract_tiles(self, fn):
        """Main method to extract and save tissue tiles"""
        tiles = []
        metadata = self.ts.getMetadata()
        
        for y in range(0, metadata['sizeY'], self.tile_size):
            for x in range(0, metadata['sizeX'], self.tile_size):
                if len(tiles) >= self.max_tiles:
                    break
                
                tissue_percent = self._get_tissue_percentage(x, y)
                if tissue_percent >= self.tissue_threshold:
                    tile_img = self.ts.getRegion(
                        region=dict(left=x, top=y, 
                                    width=self.tile_size, height=self.tile_size),
                        format=large_image.tilesource.TILE_FORMAT_NUMPY,
                        level=0
                    )[0]
                                        
                    if tile_img.size > 0:
                        if fn(tile_img):
                            tiles.append((tissue_percent, x, y, tile_img))
        
        # Save top tiles sorted by tissue percentage
        for idx, (score, x, y, img) in enumerate(sorted(tiles, key=lambda x: -x[0])):
            fname = f"{self.name}_tile_{idx}_x{x}_y{y}_score{score:.1f}.png"
            Image.fromarray(img).save(os.path.join(self.output_dir, fname))

# Usage
if __name__ == "__main__":
    extractor = TissueTileExtractor(
        slide_path='./data/patient_000_node_1.tif',
        output_dir='./data/tiles',
        tile_size=1024,
        tissue_threshold=95.0,
        max_tiles=1000
    )
    extractor.extract_tiles()



