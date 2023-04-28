# Single, Mirror and Multi-Machine Distributed Training for CIFAR-10

import tensorflow as tf
import tensorflow
from tensorflow.python.client import device_lib
import argparse
import os
import sys
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io
import hypertune

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--epochs',
      required=True,
      type=int,
      help='training epochs')
    parser.add_argument(
      '--steps_per_epoch',
      required=True,
      type=int,
      help='steps_per_epoch')
    parser.add_argument(
      '--learning_rate',
      required=True,
      type=float,
      help='learning rate')
    parser.add_argument(
      '--batch_size',
      required=True,
      type=int,
      help='training batch size')
    parser.add_argument(
      '--loss',
      required=True,
      type=str,
      help='loss function')
    
    args = parser.parse_args()
    return args

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Single Machine, single compute device
DISTRIBUTE='single'
if DISTRIBUTE == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple compute device
elif DISTRIBUTE == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif DISTRIBUTE == 'multi':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Multi-worker configuration
print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

# Preparing dataset
BUFFER_SIZE = 10000

def make_datasets_unbatched():
    # Load train, validation and test sets
    dest = 'gs://data-bucket-417812395597/'
    train_x = np.load(BytesIO(
        file_io.read_file_to_string(dest+'train_x', binary_mode=True)
    ))
    train_y = np.load(BytesIO(
        file_io.read_file_to_string(dest+'train_y', binary_mode=True)
    ))
    val_x = np.load(BytesIO(
        file_io.read_file_to_string(dest+'val_x', binary_mode=True)
    ))
    val_y = np.load(BytesIO(
        file_io.read_file_to_string(dest+'val_y', binary_mode=True)
    ))
    test_x = np.load(BytesIO(
        file_io.read_file_to_string(dest+'test_x', binary_mode=True)
    ))
    test_y = np.load(BytesIO(
        file_io.read_file_to_string(dest+'test_y', binary_mode=True)
    ))
    return train_x, train_y, val_x, val_y, test_x, test_y

def tf_model():
    black_n_white_input = tensorflow.keras.layers.Input(shape=(80, 80, 1))
    
    enc = black_n_white_input
    
    #Encoder part
    enc = tensorflow.keras.layers.Conv2D(
        32, kernel_size=3, strides=2, padding='same'
    )(enc)
    enc = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(enc)
    enc = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(enc)
    
    enc = tensorflow.keras.layers.Conv2D(
        64, kernel_size=3, strides=2, padding='same'
    )(enc)
    enc = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(enc)
    enc = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(enc)
    
    enc = tensorflow.keras.layers.Conv2D(
        128, kernel_size=3, strides=2, padding='same'
    )(enc)
    enc = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(enc)
    enc = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(enc)
    
    enc = tensorflow.keras.layers.Conv2D(
        256, kernel_size=1, strides=2, padding='same'
    )(enc)
    enc = tensorflow.keras.layers.LeakyReLU(alpha=0.2)(enc)
    enc = tensorflow.keras.layers.Dropout(0.5)(enc)
    
    #Decoder part
    dec = enc
    
    dec = tensorflow.keras.layers.Conv2DTranspose(
        256, kernel_size=3, strides=2, padding='same'
    )(dec)
    dec = tensorflow.keras.layers.Activation('relu')(dec)
    dec = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(dec)
    
    dec = tensorflow.keras.layers.Conv2DTranspose(
        128, kernel_size=3, strides=2, padding='same'
    )(dec)
    dec = tensorflow.keras.layers.Activation('relu')(dec)
    dec = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(dec)
    
    dec = tensorflow.keras.layers.Conv2DTranspose(
        64, kernel_size=3, strides=2, padding='same'
    )(dec)
    dec = tensorflow.keras.layers.Activation('relu')(dec)
    dec = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(dec)
    
    dec = tensorflow.keras.layers.Conv2DTranspose(
        32, kernel_size=3, strides=2, padding='same'
    )(dec)
    dec = tensorflow.keras.layers.Activation('relu')(dec)
    dec = tensorflow.keras.layers.BatchNormalization(momentum=0.8)(dec)
    
    dec = tensorflow.keras.layers.Conv2D(
        3, kernel_size=3, padding='same'
    )(dec)
    
    color_image = tensorflow.keras.layers.Activation('tanh')(dec)
    
    return black_n_white_input, color_image

# Build the and compile TF model
def build_and_compile_tf_model(loss_fn, learning_rate):
    black_n_white_input, color_image = tf_model()
    model = tensorflow.keras.models.Model(
        inputs=black_n_white_input,
        outputs=color_image
    )
    _optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.5
    )
    model.compile(
        loss=loss_fn,
        optimizer=_optimizer
    )
    return model

def main():
    args = get_args()
    
    NUM_WORKERS = strategy.num_replicas_in_sync
    # Here the batch size scales up by number of workers since
    # `tf.data.Dataset.batch` expects the global batch size.
    GLOBAL_BATCH_SIZE = args.batch_size * NUM_WORKERS
    MODEL_DIR = os.getenv("AIP_MODEL_DIR")

    train_x, train_y, val_x, val_y, _, _ = make_datasets_unbatched()

    with strategy.scope():
        # Creation of dataset, and model building/compiling need to be within
        # `strategy.scope()`.
        model = build_and_compile_tf_model(args.loss, args.learning_rate)

    history = model.fit(
        train_x,
        train_y,
        batch_size=GLOBAL_BATCH_SIZE,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=(val_x, val_y),
    )
    model.save(MODEL_DIR)
    
    # DEFINE HPT METRIC
    hp_metric = history.history['val_loss'][-1]

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='val_loss',
      metric_value=hp_metric,
      global_step=args.epochs)


if __name__ == "__main__":
    main()
