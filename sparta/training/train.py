import os
import json
import pandas
import argparse
from datetime import datetime
import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from segmentation_models.metrics import iou_score # make sure to import your metric functions here
from segmentation_models.losses import jaccard_loss, categorical_focal_jaccard_loss, categorical_focal_dice_loss  # make sure to import your loss functions here
from segmentation_models.losses import binary_focal_dice_loss, binary_focal_jaccard_loss, cce_jaccard_loss

from sparta.training.models import *
from sparta.training.custom import *
from sparta.tfrecords.read import *


def mkdirs(directory: str) -> None:
    """
    Makes the specified directory and any contingent directories if they do not exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(config_file: str) -> dict:
    """
    Load config file into python dictionary
    """
    with open(config_file) as f:
        config = json.load(f)
    return config

def command_config() -> dict:
    """
    Load the config file from command line into python dictionary
    """
    parser = argparse.ArgumentParser(description='Train Models')
    parser.add_argument('config', help='config file [.json]', metavar='config', type=str)
    return load_config(vars(parser.parse_args())['config'])

def get_loss(config: dict):
    """
    Return the loss function given the config
    """
    if "custom_loss_func" in config.keys():
        return globals()[config['custom_loss_func']]
    return config['loss_func']

def get_metric(config: dict):
    """
    Return the metric function given the config
    """
    if "custom_metric" in config.keys():
        return globals()[config['custom_metric']]
    return config['metric']

def load_object(val):
    """
    Loads singluar custom object, if it is a list, assumes first argument
        is a string to a callable name and the rest are parameters for the callable
    """
    if isinstance(val, list):
        return globals()[val[0]](*val[1:])
    return globals()[val]

def load_customobjects(config: dict):
    """
    Loads any custom objects needed to restore/finetune a model.
    """
    objects = config.get('custom_objects', None)
    if objects is None:
        return None
    elif isinstance(objects, str):
        try:
            dobjects = {key: load_object(val) for key, val in json.loads(objects.replace('\'', '\"')).items()}
        except:
            dobjects = None
    return dobjects

def load_datasets(config: dict):
    """
    Load data into tensorflow Dataset objects.
    """
    train_dataset, ntrain_samples = globals()[config['dataset_func']](config, train=True)
    val_dataset, nval_samples = globals()[config['dataset_func']](config, train=False)
    
    steps_per_epoch = ntrain_samples // config['batch']
    val_steps_per_epoch = nval_samples // config['batch']

    print(f"Training -> Samples: {ntrain_samples},  Targets: {ntrain_samples}")
    print(f"Validation -> Samples: {nval_samples},  Targets: {nval_samples}")
    print(f"Steps per Training Epoch -> {steps_per_epoch}")
    print(f"Steps per Validation Epoch -> {val_steps_per_epoch}")

    return train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch

def get_compiled_model(config: dict) -> tf.keras.Model:
    """
    Return compiled model for specified downstream task.
    """
    if config['optimizer'] == "rmsprop":
        if 'scheduler' in config.keys():
            opt = tf.keras.optimizers.RMSprop(learning_rate=CustomSchedule(config['embed_dim']))
        else:
            opt = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    elif config['optimizer'] == "adam":
        if 'scheduler' in config.keys():
            opt = tf.keras.optimizers.Adam(learning_rate=CustomSchedule(config['embed_dim']))
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    elif config['optimizer'] == "adamw":
        if 'scheduler' in config.keys():
            opt = tf.keras.optimizers.AdamW(learning_rate=CustomSchedule(config['embed_dim']))
        else:
            opt = tf.keras.optimizers.AdamW(learning_rate=config['learning_rate'])

    if config['restore']:
        print("Restoring Model")
        model = tf.keras.models.load_model(config['model_path'], custom_objects=load_customobjects(config))
        config['model_out'] = f"{os.path.basename(config['model_path']).split('.h5')[0]}_restored"
        model.summary()
        return model
    
    elif config['finetune']:
        print("Finetuning Model")
        model = tf.keras.models.load_model(config['model_path'], custom_objects=load_customobjects(config))
        config['model_out'] = f"{os.path.basename(config['model_path']).split('.h5')[0]}_finetuned"
        model.trainable = True
        nlayers = len(model.layers)
        flayers = round(config['freeze_perc'] * nlayers)
        for layer in model.layers[:flayers]:
            layer.trainable = False
        model.compile(
                optimizer=opt,
                loss=get_loss(config), 
                metrics=[get_metric(config)] 
                )
        model.summary()
        return model

    print("Compiling and returning model")
    model = globals()[config['model']](config)
    model.summary()
    model.compile(
        optimizer=opt,
        loss=get_loss(config), 
        metrics=[get_metric(config)] 
        )
    config['model_out'] = config['model']
    return model

def strategy_compile(config: dict) -> tf.keras.Model:
    """
    If multiple GPUs will invoke MirroredStrategy to train across them (only multi-GPU on single node).
    """
    devices = tf.config.experimental.list_physical_devices('GPU')
    devices_names = [d.name.split("e:")[1] for d in devices]
    print(devices_names)
    if len(devices) > 0:
        strategy = tf.distribute.MirroredStrategy(devices=devices_names[:len(devices)])
        with strategy.scope():
            model = get_compiled_model(config)
    else:
        model = get_compiled_model(config)
    return model

def set_callbacks(config: dict) -> list:
    """
    Function to invoke all training callbacks you want to use.
    """
    today = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    callbacks = {
        "model_output": tf.keras.callbacks.ModelCheckpoint(f"{config['output']}/model/{config['model_out']}_{today}.h5", 
                                                           monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False),
        "csv_logger": tf.keras.callbacks.CSVLogger(f"{config['output']}/{config['model_out']}_{today}.csv", separator=',', append=True),
        "write_out": WriteConfig(config, f"{config['output']}/{config['model_out']}_{today}.csv", f"{config['output']}/{config['model_out']}_{today}.json"),
        "reduce_lr": tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=config['patience'] if 'patience' in config.keys() else 4),
        "tensorboard": tf.keras.callbacks.TensorBoard(log_dir=config['output'], profile_batch=(500,550), write_images=True, update_freq='epoch', histogram_freq=1)
    }
    return [callbacks[key] for key in config['callbacks']]

def run_training(config: dict) -> None:
    """
    Main training function called to train your model and track your experiment.
    """
    tf.random.set_seed(config['seed'])
    mkdirs(f"{config['output']}/model/logs")
    print("Loading data")
    with tf.device('/cpu:0'):
        train_dataset, val_dataset, steps_per_epoch, val_steps_per_epoch = load_datasets(config)
    model = strategy_compile(config)
    print(f"Training model: {config['model_out']}")
    if isinstance(train_dataset, tuple):
        print("Dataset is a tuple")
        _ = model.fit(
            train_dataset[0],
            train_dataset[1],
            batch_size=config['batch'],
            epochs=config['epochs'],
            validation_data=(val_dataset[0], val_dataset[1]),
            callbacks=set_callbacks(config)
    )
    else:
        print('Using tf.Dataset')
        _ = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=config['epochs'],
            validation_data=val_dataset,
            validation_steps=val_steps_per_epoch,
            callbacks=set_callbacks(config)
        )

if __name__ == '__main__':
    run_training(command_config())
