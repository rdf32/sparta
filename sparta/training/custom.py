import json
import numpy as np
import pandas as pd
import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss

class AugmentSegmentation(tf.keras.layers.Layer):
    """
    Applies data augmentation to training data, additional augmentations can be added but must
    be conscious of which ones are being applied in segmentation tasks (only specific ones work)
    for more methods go here https://www.tensorflow.org/tutorials/images/data_augmentation
    """
    def __init__(self):
        super().__init__()
    def call(self, inputs, labels):
        choice = np.random.choice([0, 1])
        return tf.experimental.numpy.flip(inputs, axis=choice), tf.experimental.numpy.flip(labels, axis=choice)
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Creates wawrmup learning rate schedule for transformers
    """
    def __init__(self, d_model, warmup_steps=5000):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(tf.cast(self.d_model, tf.float32)) * tf.math.minimum(arg1, arg2)
    def get_config(self):
        config = {
        'd_model': self.d_model,
        'warmup_steps': self.warmup_steps}
        return config
    
class WriteConfig(tf.keras.callbacks.Callback):
    """
    Write out training loss/metric scores to csv file at the end of each epoch.
    """
    def __init__(self, config, csv, config_out):
        super(WriteConfig, self).__init__()
        self.config = config
        self.csv = csv
        self.config_out = config_out

    def on_epoch_end(self, epoch, logs=None):
        df = pd.read_csv(self.csv)
        self.config.update(df.iloc[int(df[['val_loss']].idxmin().iloc[0])])
        with open(self.config_out, 'w') as outfile:
            json.dump(self.config, outfile, indent=4)

def masked_categorical_focal_loss(label, pred, mval=-1):
    loss_object = SparseCategoricalFocalLoss(2, reduction='none')
    loss = loss_object(tf.boolean_mask(label, tf.not_equal(label, mval)), tf.boolean_mask(pred, tf.not_equal(label, mval)))
    return tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(label != mval, dtype=loss.dtype))

def masked_accuracy(label, pred, mval=-1):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != mval

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


