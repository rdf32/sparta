import os
import json
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.coords import BoundingBox
from tqdm import tqdm
from dataclasses import dataclass

import sys
import multiprocessing as mp
from typing import Union

ard = gpd.read_file('./conus_ard/conus_c2_ard_grid.shp')

class_maps = {
    "output": {
        'water': 11, 'snow': 12, 'open': 21, 'low': 22, 'med': 23, 'high': 24, 'barren': 31, 'dforest': 41, 'eforest': 42,
        'mforest': 43, 'tshrub': 45, 'tgrass': 46, 'shrub': 52, 'grassland': 71, 'hay': 81, 'crops': 82, 'wwetlands': 90,'ewetlands': 95
        }
}
pred_class = {val:ind for ind, val in enumerate(class_maps['output'].keys())}
convert_map = {val:ind for ind, val in enumerate(class_maps['output'].values())}

@dataclass
class Tile:
    h: int
    v: int

@dataclass
class Bounds:
    bounds: BoundingBox
    h: int
    v: int

@dataclass
class Block:
    h: int
    v: int
    xgeo: float
    ygeo: float
    ardrow: int
    ardcol: int


def tile_bounds(h: int, v: int) -> tuple:
    """
    Return the bounds of a given ard tile
    """
    gdf = ard.loc[(ard['h'] == (h)) & (ard['v'] == v)]
    return rio.coords.BoundingBox(*gdf.unary_union.bounds)

def train_val_split(samples, percent: float, random: int):
    """
    Given samples, randomly select indexes for training and validation
    """
    indx = np.arange(len(samples)).astype(int)
    np.random.seed(random)
    np.random.shuffle(indx)
    return np.array(indx[:int(percent*len(indx))]), np.array(indx[int(percent*len(indx)):])

def read_image(path: str, xgeo: int, ygeo: int, ncols: int, nrows: int) -> np.ndarray:
    """
    Read raster window into numpy array.
    """
    with rio.open(path) as ds:
        row, col = ds.index(xgeo, ygeo)
        data = ds.read(
            window=rio.windows.Window(col, row, ncols, nrows)
            )
        data[data == ds.nodata] = 0
    return data.squeeze()

def read_image_bounds(path: str, bounds: rio.coords.BoundingBox):
    """
    Reads an image path into a nodata masked array and returns the data along with geographic metadata.
    """
    with rio.open(path) as ds:
        l, b, r, t = bounds
        array = ds.read(
            window=rio.windows.from_bounds(l, b, r, t, ds.transform),
            masked=True, boundless=True, fill_value=ds.nodata
        )
        profile = ds.profile
        profile.update({'width': array.data.shape[2], 'height': array.data.shape[1], 
                        'transform': rio.transform.from_bounds(l, b, r, t, array.data.shape[2], array.data.shape[1])})
    return array.data, profile

def read_image_offsets(path: str, xgeo: int, ygeo: int, ncols: int, nrows: int, coloff: int, rowoff: int) -> np.ndarray:
    """
    Read raster window into numpy array.
    """
    with rio.open(path) as ds:
        row, col = ds.index(xgeo, ygeo)
        data = ds.read(
            boundless=True,
            window=rio.windows.Window(col - coloff, row - rowoff, ncols, nrows)
            )
        data[data == ds.nodata] = 0
    return data.squeeze() 

def scale_raster(array: np.ndarray, collection: str) -> np.ndarray:
    """
    Apply proper scaling for either collection-1 or collection-2 data
    """
    if collection == 'collection-1':
        return array * 0.0001
    elif collection == 'collection-2':
        return np.ndarray.clip((array * 0.0000275) - 0.2, 0, np.max(array))
    
def landcover_classes(array: np.ndarray, cmap) -> np.ndarray:
    """
    Converts class labels from arbitrary labeling integers to 0-nclasses
    """
    return np.vectorize(lambda x: cmap[int(x)])(array)

def pool_map(func, vals) -> list:
    """
    Use a pool of workers to do some work
    """
    cpu = mp.cpu_count()
    out = []
    with mp.Pool(cpu) as pool:
        with tqdm(total=len(vals), file=sys.stdout) as pbar:
            for res in pool.imap(func, vals):
                out.append(res)
                pbar.update()
    return out

def get_proj_rio(path: str) -> rio.CRS:
    """
    Return projection of raster in the flavor of rasterio
    """
    with rio.open(path) as src:
        crs = src.crs
    return crs

def output_meta(affine, proj):
    meta = {'driver': 'COG',
            'dtype': 'uint8',
            'nodata': None,
            'width': 5000,
            'height': 5000,
            'count': 1,
            'crs': proj,
            'compress': 'deflate', # deflate instead of lzw
            'transform': affine,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256}
    return meta

def write_gtiff(oarray: np.ndarray, path: str, meta: dict, cmap: dict = None):
    with rio.Env():
        with rio.open(path, 'w', **meta) as dst:
            dst.write(np.expand_dims(oarray, axis=0))
            if cmap is not None:
                dst.write_colormap(1, cmap)
    return True

def mkdirs(directory: str) -> None:
    """
    Makes the specified directory and any contingent directories if they do not exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def encode_file(key: str, path: str, params: dict) -> Union[str, list]:
    """
    Add the proper sequence of years to input data from config
    """
    if key in ['impervious', 'landcover', 'descriptor']:
        return [path.replace('{year}', str(year)) for year in params['years']]
    elif key in ['leafon', 'leafoff']:
        return [path.replace('{year}', str(year)) for year in range(params['min_year'], params['max_year'] + 1)]
    elif key in ['mask']:
        return path.replace('{hvblock}', str(params['hvblock']))
    else:
        return path

def load_files(config: dict, params: dict) -> dict:
    """
    Load files from config into dictionary for data loading
    """
    return {key: encode_file(key, path, params) for key, path in config['spectral']['paths'].items()}

def average_ensemble(models: list, data: np.ndarray) -> np.ndarray:
    """
    Given a list of models and some data, return average prediction
    """
    return np.sum(np.array([model(data, training=False).numpy().squeeze() for model in models]), axis=0) / len(models)

def load_json(path: str) -> dict:
    """
    Load something from a gzip json file
    """
    if not os.path.exists(path):
        return []
    with open(path, 'rt') as gf:
        return json.load(gf)
    
def load_config(config_file: str) -> dict:
    """
    Load config file into python dictionary
    """
    with open(config_file) as f:
        config = json.load(f)
    return config

def load_data(func, files: dict) -> dict:
    """
    Run specified function on paths in files dictionary, used for loading data into memory
    """
    return {key: [func(file) for file in val] if isinstance(val, list) else func(val) for key, val in files.items()}

def get_hvblock(h: int, v: int, config: dict) -> Union[None, str]:
    """
    Given and ARD tile h, v returns which hvblock (refinement block) is belongs to
    """
    for hvblock, sblocks in load_json(config['blocks']).items():
        if [h, v] in sblocks:
            print(h,v, f"block: {hvblock}")
            return hvblock
    return None
