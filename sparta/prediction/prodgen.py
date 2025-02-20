import json
import zarr
import numpy as np
import rasterio as rio
from tqdm import tqdm
from sparta.common import read_image_bounds, class_maps, mkdirs

output_map = {ind: val for ind, val, in enumerate(class_maps['output'].values())}

def mapclass(indices: np.ndarray, class_map: dict) -> np.ndarray:
    """
    Look up the class value using the index key
    """
    return np.vectorize(lambda x: class_map[int(x)])(indices)

def output_meta(height, width, count, crs, affine, dtype, nodata):
    meta = {'driver': 'COG',
            'dtype': dtype,
            'nodata': nodata,
            'width': width,
            'height': height,
            'count': count,
            'crs': crs,
            'compress': 'deflate', # deflate instead of lzw
            'transform': affine,
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256}
    return meta

def nodata_mask(bounds: rio.coords.BoundingBox):
    nlcd_array, _ = read_image_bounds("./NLCD_Land_Cover.tif", bounds)
    return nlcd_array.squeeze() == 0

def write_gtiff(oarray, path, meta, cmap=None):
    with rio.Env():
        with rio.open(path, 'w', **meta) as dst:
            dst.write(np.expand_dims(oarray, axis=0))
            if cmap is not None:
                dst.write_colormap(1, cmap)
    return True

def load_cmap(cover):
    cmaps = {
        'Land_Cover': './landcover_colors.json',
        'Science_Product': './science_product_colors.json'
    }
    try:
        with open(cmaps[cover], 'r') as f:
            cmap = json.load(f)
            cmap = {int(key): tuple(val) for key, val in cmap.items()}
        return cmap
    except:
        return None

def get_path(h, v, min_year, max_year, version, pred_dir):
    return f"{pred_dir}/{version}_h{h:02}v{v:02}_{min_year}_{max_year}.zarr"

def get_affine(bounds):
    return rio.transform.Affine.from_gdal(bounds.left, 30, 0, bounds.top, 0, -30)

def collapse_transition(array):
    orray = array.copy()
    orray[orray == 45] = 52
    orray[orray == 46] = 71
    return orray


def generate_products(h, v, min_year, max_year, pred_dir, version):
    mkdirs(f"{pred_dir}/science_product")
    mkdirs(f"{pred_dir}/landcover")
    
    years = {year: ind for ind, year in enumerate(range(min_year, max_year + 1))}
    zarray = zarr.open(get_path(h, v, min_year, max_year, version, pred_dir))
    preds = mapclass(np.argmax(zarray, axis=1), output_map) 
    meta =  output_meta(preds.shape[1], preds.shape[2], 1, rio.crs.CRS.from_wkt(zarray.attrs['projection']),
                        get_affine(rio.coords.BoundingBox(*zarray.attrs['bounds'])), np.uint8, 255)
    mask = nodata_mask(rio.coords.BoundingBox(*zarray.attrs['bounds']))
    
    for year, ind in tqdm(years.items(), "generating science products..."):
        science_path = f"{pred_dir}/science_product/{version}_h{h:02}v{v:02}_{year}.tif"
        oarray = preds[ind].astype(np.uint8)
        oarray[mask] = 255
        write_gtiff(oarray, science_path, meta, load_cmap('Science_Product'))
        
    for year, ind in tqdm(years.items(), "generating land cover...."):
        land_path = f"{pred_dir}/landcover/{version}_h{h:02}v{v:02}_{year}.tif"
        oarray = collapse_transition(preds[ind].astype(np.uint8))
        oarray[mask] = 255
        write_gtiff(oarray, land_path, meta, load_cmap('Land_Cover'))
        
    return True

if __name__ == '__main__':
    root = './predictions'
    min_year, max_year = 1984, 2021

    hvs = [(4, 1), (13, 13), (20, 8), (24, 13), (3, 10)]
    versions = ['spatial', 'transformer_encoder', 'transformer_encoder_mlp', 'sparta_transformer_encoder', 'sparta_transformer_encoder_mlp'] 

    for h, v in hvs:
        for version in versions:
            generate_products(h, v, min_year, max_year, root, version)