import argparse
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from functools import partial

from sparta.common import convert_map, pool_map, read_image
from sparta.common import scale_raster, landcover_classes, mkdirs

from sparta.tfrecords.write import samples_records, chip_example

def landcover(sample, tile, scaling):
    datas = {data: read_image(path=sample[data], xgeo=sample['xgeo'], ygeo=sample['ygeo'], ncols=tile, nrows=tile) 
            for data in ['image', 'target']}
    return (scale_raster(datas['image'], scaling).transpose(),
            landcover_classes(datas['target'].squeeze().transpose(), convert_map))

def create_records(i, data, path, nsamples):
    samples = data.iloc[(i * nsamples) : ((i + 1) * nsamples)]
    with tf.io.TFRecordWriter(
        f"{path}" + "/file_%.2i-%i.tfrec" % (i, len(samples))
    ) as writer:
        for i in tqdm(range(len(samples)), desc="inner"):
            image, mask = landcover(samples.iloc[i], 256, 'collection-1')
            example = chip_example(image, mask)
            writer.write(example.SerializeToString())

def command_line():
    """
    Read which h, v ard tile from command line
    """
    parser = argparse.ArgumentParser(description='Create U-net Tfrecords')
    parser.add_argument('outdir', help='outdir', metavar='outdir', type=str)
    args = vars(parser.parse_args())
    return args['outdir']

if __name__ == '__main__':

    out_dir = command_line()
    mkdirs(f"{out_dir}/unet/train")
    mkdirs(f"{out_dir}/unet/val")

    print('training samples')
    traindf = pd.read_csv('./samples/train_unet_chips_256.csv')
    tnsamples, tnrecords = samples_records(traindf, n=20)
    _ = pool_map(partial(create_records, data=traindf, 
                         path=f"{out_dir}/unet/train", nsamples=tnsamples), range(tnrecords))

    print('validation samples')    
    valdf = pd.read_csv('./samples/val_unet_chips_256.csv')

    vnsamples, vnrecords = samples_records(valdf, n=20)
    _ = pool_map(partial(create_records, data=valdf, 
                         path=f"{out_dir}/unet/val", nsamples=vnsamples), range(vnrecords))
