import zarr
import time
import argparse
import numpy as np
from functools import partial
from rasterio.coords import BoundingBox
from typing import Tuple, Generator

import tensorflow as tf
from sparta.common import mkdirs, scale_raster
from sparta.common import read_image_offsets, get_proj_rio, load_files
from sparta.common import load_data, load_config, tile_bounds, Bounds, Block
from sparta.training.models import *
from mpiframework import MPI_PROCESS, user_defined, run_time, chunk_gen
    
@user_defined
def create_zarr(outpaths: dict, profile: dict): #bounds, projection, gsd, min_year, max_year, models
    """
    Initialization function to create the required output zarr arrays
    """
    mkdirs(f"{profile['output']}/predictions")
    cols = int((profile['bounds'].bounds.right - profile['bounds'].bounds.left) / 30) 
    rows = int((profile['bounds'].bounds.top - profile['bounds'].bounds.bottom) / 30)
    outchunk = tuple(profile['outshape'])

    slzarray = zarr.open(outpaths[profile['model']],
                mode='w',
                shape=(outchunk[0], outchunk[1], rows, cols),
                chunks=outchunk,
                dtype='float32',
                write_empty_chunks=False,
                fill_value=profile['nodata'])
    slzarray.attrs['bounds'] = profile['bounds'].bounds
    slzarray.attrs['projection'] = profile['projection']
    
    return {}

@user_defined
def my_result(result: dict, ttime: list) -> None:
    """
    Adds compute time for each component to kwargs variable
    """
    ttime.append(result['ttime'])


@user_defined
def my_final(start: float, ttime: list) -> None:
    """
    Prints final compute times of each component at the end of processing
    """
    print("Ttime compute time: ", np.sum(ttime))
    print("Total compute time: ", time.time() - start)

@user_defined
def my_chunks(profile: dict, gsd=30, sub=0) -> Tuple[int, Generator]: # htile is size of parquet files so here 500
    mchunk = profile['mchunk']
    total_chunks = []
    for ardrow in range(0, int((profile['bounds'].bounds.top - profile['bounds'].bounds.bottom) / gsd), mchunk):
        for ardcol in range(0, int((profile['bounds'].bounds.right - profile['bounds'].bounds.left) / gsd), mchunk):
            total_chunks.append(
                Block(
                    profile['bounds'].h, # h
                    profile['bounds'].v, # v
                    (ardcol * gsd) + profile['bounds'].bounds.left, # xgeo
                    profile['bounds'].bounds.top - (ardrow * gsd), # ygeo
                    ardrow, # ard row
                    ardcol)) # ard col
    return len(total_chunks[sub:]), chunk_gen(total_chunks[sub:])

@user_defined
def my_worker(chunk: Block, inpaths: dict, outpaths: dict, profile: dict) -> dict:
    """
    Generates segment based landcover and impervious predictions, leveraging spatial sequences, temporal segment information and 
        timeseries modeling (transformers) to create a spectral spatiotemporal prediction following the LCAMS legend
    """
    try:
        return predict(chunk, inpaths, outpaths, profile)
    except:
        print("failed pred", chunk)
        return {"spatialtime": 0, "refinementtime": 0, "impervioustime": 0}

def predict(chunk: Block, inpaths: dict, outpaths: dict, profile: dict) -> dict:
    """
    Perform impervious fractional cover, tier-1 into tier-2 classification and
        fill any pixels where the entire timeseries is null, writes both impervious
        and classification to appropriate zarr arraysFncols
    """
    mchunk = profile['mchunk']

    datas = load_data(partial(read_image_offsets, xgeo=chunk.xgeo, ygeo=chunk.ygeo, ncols=mchunk, nrows=mchunk, coloff=0, rowoff=0), # 256 is unet shape
            {file: inpaths[file] for file in ['leafon']})
    input_data = scale_raster(np.array(datas['leafon']), profile['scaling']['leaf_on'])
    tprobs, ttime = run_time(predict_transformer)(profile['models']['tformer']['all'][0], input_data, mchunk, profile)
    output = zarr.open(outpaths[profile['model']], 'a')
    output[:, :, chunk.ardrow: chunk.ardrow + mchunk, chunk.ardcol: chunk.ardcol + mchunk] = tprobs
    
    return {"ttime": ttime}


def predict_transformer(model: tf.keras.Model, datacube: np.ndarray, mchunk: int, profile: dict) -> np.ndarray:

    inputs = np.moveaxis(datacube.reshape(datacube.shape[0], datacube.shape[1], -1), -1, 0)
    if inputs.shape[1] > profile['train_seqlen']:
        oarray = np.full((inputs.shape[0], inputs.shape[1], 18), profile['nodata'], dtype=np.float32)
        for i in range(inputs.shape[1] - profile['train_seqlen'] + 1):
            preds = model(inputs[:, i:profile['train_seqlen']+i, :], training=False)
            if i == 0:
                oarray[:, :profile['train_seqlen'], :] = preds
            else:
                oarray[:, profile['train_seqlen']+i-1, :] = preds[:, -1, :]
        oarray = np.moveaxis(oarray, 0, -1)
        return oarray.reshape(datacube.shape[0], 18, mchunk, mchunk)

def command_config_h_v_years_mchunk() -> Tuple[int, int]:
    """
    Read which h, v ard tile from command line
    """
    parser = argparse.ArgumentParser(description='Train unet model')
    parser.add_argument('config', help='config file [.json]', metavar='config', type=str)
    parser.add_argument('h', help='h', metavar='h', type=str)
    parser.add_argument('v', help='v', metavar='v', type=str)
    parser.add_argument('min_year', help='min_year', metavar='min', type=str)
    parser.add_argument('max_year', help='max_year', metavar='max', type=str)
    parser.add_argument('mchunk', help='mchunk', metavar='chunk', type=str)
    args = vars(parser.parse_args())
    return load_config(args['config']), int(args['h']), int(args['v']), int(args['min_year']), int(args['max_year']), int(args['mchunk'])

if __name__ =='__main__':
    config, h, v, min_year, max_year, mchunk = command_config_h_v_years_mchunk()
    config['max_year'] = max_year
    config['min_year'] = min_year
    inpaths = load_files(config, {"min_year": config['min_year'], "max_year": config['max_year']})

    outpaths = {output: f"{config['pred_output']}/predictions/{output}_h{h:02}v{v:02}_{config['min_year']}_{config['max_year']}.zarr" 
        for output in [config['model']]}

    config['models']['tformer']['all'] = [tf.keras.models.load_model(path, custom_objects={"PositionalEmbedding": PositionalEmbedding(embed_dim=6)}, compile=False) for path in config['models']['tformer']['all']]
    config['projection'] = get_proj_rio().to_wkt()
    config['bounds'] = Bounds(BoundingBox(*tile_bounds(h, v)), h, v)
    config['mchunk'] = mchunk
    config['outshape'] = [max_year - min_year + 1, 18, mchunk, mchunk]


    process = MPI_PROCESS(
        my_chunks,
        my_worker,
        my_result,
        initfunc=create_zarr,
        finalfunc=my_final,
        inpaths=inpaths,
        outpaths=outpaths, 
        profile=config,
        start=time.time(), 
        ttime=[]
    )
    process()






