# Single, Mirror and Multi-Machine Distributed Training

import tensorflow as tf
import tensorflow
from tensorflow.python.client import device_lib
import argparse
import os
import sys
from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io

# parse required arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', dest='lr',
                    default=0.001, type=float,
                    help='Learning rate.')
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')
parser.add_argument('--steps', dest='steps',
                    default=35, type=int,
                    help='Number of steps per epoch.')
parser.add_argument('--distribute', dest='distribute', type=str, default='single',
                    help='distributed training strategy')
args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
print('DEVICES', device_lib.list_local_devices())

# Single Machine, single compute device
if args.distribute == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple compute device
elif args.distribute == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif args.distribute == 'multi':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Multi-worker configuration
print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

# Preparing dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 128

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
def build_and_compile_tf_model():
    black_n_white_input, color_image = tf_model()
    model = tensorflow.keras.models.Model(
        inputs=black_n_white_input,
        outputs=color_image
    )
    _optimizer = tensorflow.keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5
    )
    model.compile(
        loss='mse',
        optimizer=_optimizer
    )
    return model

# Train the model
NUM_WORKERS = strategy.num_replicas_in_sync
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS
MODEL_DIR = os.getenv("AIP_MODEL_DIR")

train_x, train_y, _, _, _, _ = make_datasets_unbatched()

with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    model = build_and_compile_tf_model()

model.fit(
    train_x,
    train_y,
    epochs=args.epochs,
    steps_per_epoch=args.steps
)
model.save(MODEL_DIR)
