import random
import numpy as np
import tensorflow as tf
from typing import Tuple
from functools import partial

from sparta.training.custom import AugmentSegmentation
segmentation_augmentation = AugmentSegmentation()   

## reading tfrecords
def tfrecord(config: dict, train: bool = True):
    """
    Function for loading tfrecord datasets into tensorflow Datasets.
    """
    files, nsamples = load_records(config, train)
    dataset = (
        tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        .shuffle(config['batch'] * 10)
        .map(globals()[config['parse_func']], num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config['batch'], drop_remainder=True).repeat(config['epochs'])
    )
    if train:
        if 'augmentation_func' in config.keys():
            dataset = dataset.map(globals()[config['augmentation_func']], num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, nsamples

def tfrecord_onehot(config: dict, train: bool = True) -> Tuple[tf.data.Dataset, int]:
    """
    Function for loading tfrecord datasets into tensorflow Datasets.
    """
    files, nsamples = load_records(config, train)
    dataset = (
        tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        .shuffle(config['batch'] * 10)
        .map(globals()[config['parse_func']], num_parallel_calls=tf.data.AUTOTUNE)
        .map(partial(globals()[config['sample_func']], n_classes=config['n_classes']), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(config['batch'], drop_remainder=True).repeat(config['epochs'])
    )
    if train:
        if 'augmentation_func' in config.keys():
            dataset = dataset.map(globals()[config['augmentation_func']], num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, nsamples

def load_records(config: dict, train: bool = True):
    """
    Loads tfrecords into list to be loaded into a tesorflow Dataset.
    """
    random.seed(config['seed'])
    if train:
        if isinstance(config['train_root'], list):
            ofiles, count = [], 0
            for path in config['train_root']:
                count += count_records(path)
                ofiles.extend(tf.io.gfile.glob(f"{path}/*.tfrec"))
                random.shuffle(ofiles)
            return ofiles, count
        ofiles = tf.io.gfile.glob(f"{config['train_root']}/*.tfrec")
        random.shuffle(ofiles)
        return ofiles, count_records(config['train_root'])
    else:
        if isinstance(config['val_root'], list):
            ofiles, count = [], 0
            for path in config['val_root']:
                count += count_records(path)
                ofiles.extend(tf.io.gfile.glob(f"{path}/*.tfrec"))
                random.shuffle(ofiles)
            return ofiles, count
        ofiles = tf.io.gfile.glob(f"{config['val_root']}/*.tfrec")
        random.shuffle(ofiles)
        return ofiles, count_records(config['val_root'])

def count_records(directory: str, total: int = 0) -> int:
    """
    Given a directory of tfrecords, returns the sample count
    """
    for file in tf.io.gfile.glob(f"{directory}/*.tfrec"):
        total += int(file.split('-')[-1].rstrip('.tfrec'))
    return total

def chip_extract(example: tf.train.Example) -> dict:
    """
    Parses the chip tfrecord and converts the data back to a tensor
    """
    feature_description = {
        "cols": tf.io.FixedLenFeature([], tf.int64),
        "rows": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "chip": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example, feature_description)

def chip_parse(example: tf.train.Example) -> dict:
    """
    TfRecord parsing function used to load leafon, leafoff, dem records.
    """
    example = chip_extract(example)
    return {"chip": tf.reshape(tf.io.decode_raw(example['chip'], tf.float32), [example['rows'], example['cols'], example['channels']]),
            "mask": tf.reshape(tf.io.decode_raw(example['mask'], tf.float32), [example['rows'], example['cols']])}

def chip_one_hot(features: dict, n_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    One-hot encode mask for chip based features
    """
    return features['chip'], tf.one_hot(tf.cast(features['mask'], tf.int32), n_classes)

def sequence_extract(example: tf.train.Example) -> dict:
    """
    Parses the point_sequence tfrecord and converts the data back to a tensor
    """
    feature_description = {
        "ET": tf.io.FixedLenFeature([], tf.int64),
        "C": tf.io.FixedLenFeature([], tf.int64),
        "c": tf.io.FixedLenFeature([], tf.string), 
        "y": tf.io.FixedLenFeature([], tf.string)
    }
    return tf.io.parse_single_example(example, feature_description)

def sequence_parse(example: tf.train.Example) -> dict:
    """
    TfRecord parsing function used to load leafon, leafoff, dem records.
    """
    example = sequence_extract(example)
    sequence = tf.reshape(tf.io.decode_raw(example['c'], tf.float32), (example['ET'], example['C']))
    # targets = tf.reshape(tf.io.decode_raw(example['y'], tf.float32), (example['ET']))
    targets = tf.io.decode_raw(example['y'], tf.float32)
    return sequence, targets


