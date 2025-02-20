import argparse
import numpy as np
from tqdm import tqdm
import rasterio as rio

import tensorflow as tf
from functools import partial
from sparta.tfrecords.write import sequence_example, samples_records
from sparta.common import Tile, convert_map, tile_bounds, scale_raster
from sparta.common import read_image_bounds, pool_map, mkdirs, landcover_classes

train_paths = {
    "image": "./leafon{year}.tif",
    "target": "./NLCD_{year}_Land_Cover.tif",
   }

train_sequence = ['2001', '2001', '2001', '2004', '2004', '2006', '2006', '2008', '2008', '2008', '2011', '2011', '2013', '2013', '2013', '2016', '2016', '2016', '2019']

def create_records(i, data, path, nsamples):
    samples = data[(i * nsamples) : ((i + 1) * nsamples)]
    with tf.io.TFRecordWriter(
        f"{path}" + "/file_%.2i-%i.tfrec" % (i, len(samples))
    ) as writer:
        for i in tqdm(range(len(samples)), desc="inner"):
            example = sequence_example(samples[i])
            writer.write(example.SerializeToString())

def load_samples(tile, train, val):
    train_samples = []
    val_samples = []
    bounds = tile_bounds(tile.h, tile.v)
    datas = {
        "image_t": np.stack([read_image_bounds(train_paths['image'].replace('{year}', str(year)), bounds)[0] for year in range(2001, 2020)], axis=0).astype(np.float32),
        "target_t": np.concatenate([read_image_bounds(train_paths['target'].replace('{year}', year), bounds)[0] for year in train_sequence]).astype(np.uint8),
        "mask": rio.open(f'./samples/h{tile.h:02}v{tile.v:02}_sample_points.tif').read(1)
        }
    tseq = datas['image_t'][..., datas['mask'] == 1]
    ttarg = datas['target_t'][..., datas['mask'] == 1]
    tvmask = ttarg[0] != 0

    vseq = datas['image_t'][..., datas['mask'] == 2]
    vtarg = datas['target_t'][..., datas['mask'] == 2]
    vvmask = vtarg[0] != 0

    train_seq = scale_raster(tseq[..., tvmask], 'collection-1').astype(np.float32)
    train_targ = landcover_classes(ttarg[..., tvmask], convert_map).astype(np.uint8)


    val_seq = scale_raster(vseq[..., vvmask], 'collection-1').astype(np.float32)
    val_targ = landcover_classes(vtarg[..., vvmask], convert_map).astype(np.uint8)
    for sample in range(train_seq.shape[-1]):
        train_samples.append({"data": train_seq[:, :, sample], "target": train_targ[:, sample]})
    for sample in range(val_seq.shape[-1]):
        val_samples.append({"data": val_seq[:, :, sample], "target": val_targ[:, sample]})

    np.random.seed(42)
    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)
    if val and train:
        return train_samples, val_samples
    if val and not train:
        return val_samples
    if train and not val:
        return train_samples

def command_line():
    """
    Read which h, v ard tile from command line
    """
    parser = argparse.ArgumentParser(description='Train unet model')
    parser.add_argument('h', help='h', metavar='h', type=str)
    parser.add_argument('v', help='v', metavar='v', type=str)
    parser.add_argument('part', help='part', metavar='part', type=str)
    parser.add_argument('outdir', help='outdir', metavar='outdir', type=str)


    args = vars(parser.parse_args())
    return int(args['h']), int(args['v']), args['part'], args['outdir']

if __name__ == '__main__':
    h, v, part, outdir = command_line()
    tile = Tile(h, v)
    mkdirs(f"{outdir}/transformer/train/h{h:02}v{v:02}")
    mkdirs(f"{outdir}/transformer/val/h{h:02}v{v:02}")

    if part == 'train':
        train_samples = load_samples(tile, True, False)
        tnsamples, tnrecords = samples_records(train_samples, n=100)
        print(tnsamples, tnrecords)
        _ = pool_map(partial(create_records, data=train_samples, path=f"{outdir}/transformer/train/h{h:02}v{v:02}", nsamples=tnsamples), range(tnrecords))
    
    elif part == 'val':
        val_samples = load_samples(tile, False, True)
        vnsamples, vnrecords = samples_records(val_samples, n=300)
        print(vnsamples, vnrecords)
        _ = pool_map(partial(create_records, data=val_samples, path=f"{outdir}/transformer/val/h{h:02}v{v:02}", nsamples=vnsamples), range(vnrecords))