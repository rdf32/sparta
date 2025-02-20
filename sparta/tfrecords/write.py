import numpy as np
import tensorflow as tf

def array_feature(array: np.ndarray) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[array.astype(np.float32).tobytes()])
    )

def int_feature(target: int) -> tf.train.Feature:
    """Returns a int64_list from a int64 single point value."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[target])
    )

def samples_records(data, n=10):
    """
    Return the number of samples and number of tfrecords needed
    """
    nsamples = len(data) // n
    nrecords = len(data) // nsamples
    if len(data) % nsamples:
        nrecords += 1
    return nsamples, nrecords

def record_example(feature: dict) -> tf.train.Example:
    """
    Creates a tfrecord given a defined feature
    """
    return tf.train.Example(features=tf.train.Features(feature=feature))

def chip_example(chip: np.ndarray, mask: np.ndarray) -> tf.train.Example:
    """
    Creates a tfrecord example for chip based data (semantic segmentation)
    """
    feature = {
        "cols": int_feature(chip.shape[0]),
        "rows": int_feature(chip.shape[1]),
        "channels": int_feature(chip.shape[2]),
        "chip": array_feature(chip),
        "mask": array_feature(mask)
    }
    return record_example(feature)

def sequence_example(example: dict) -> tf.train.Example:
    """
    Creates a tfrecord example for sequence based data (transformer)
    """
    feature = {

        "ET": int_feature(example['data'].shape[0]),
        "C": int_feature(example['data'].shape[1]),
        "c": array_feature(example['data']),
        "y": array_feature(example['target'])

    }
    return record_example(feature)


