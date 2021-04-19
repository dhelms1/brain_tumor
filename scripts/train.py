import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import argparse
import os
import json
AUTOTUNE = tf.data.AUTOTUNE
TRAIN_BUFFER = 2296

from model import EfficientNetClassifier

def tfrecord_reader(record):
    '''
    Parse a single tfrecord and return the image (formatted as a tensor)
    and the label (formatted as an integer).
    '''
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.io.parse_single_example(record, features)
    label = tf.cast(parsed['label'], tf.int32)
    return tf.io.parse_tensor(parsed['image_raw'], tf.uint8), tf.one_hot(label, 4)

def load_data(data_dir):
    '''
    Create a TFRecordDataset and decode a given tfrecord file.
    '''
    data = tf.data.TFRecordDataset(data_dir)
    data = data.map(tfrecord_reader)
    return data

def create_dataset(data_dir, BATCH_SIZE, train=False):
    '''
    Create and parse the tfrecord file for a given dataset. Turn dataset
    into batches of specified size.
    '''
    dataset = load_data(data_dir)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    if train:
        dataset = dataset.shuffle(TRAIN_BUFFER)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def model(train_dataset, val_dataset, class_weights, epochs):
    '''
    Create a TensorFlow model and train/validate on the given data. Returns
    the final model.
    '''
    model = EfficientNetClassifier()
        
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1, 
                                     factor=0.2, min_lr=0.000001)
    
    model.fit(x=train_dataset,
              epochs=epochs, 
              validation_data=val_dataset,
              verbose=2,
              class_weight=class_weights,
              callbacks=[early_stop, lr_reduction])
    
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for dataset (default=32)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='input epochs for training (default=15)')
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = parser.parse_args()
    
    # Create training and validation datasets
    train_dir = os.path.join(args.data_dir, 'train_images.tfrecords')
    val_dir = os.path.join(args.data_dir, 'val_images.tfrecords')
    
    train_dataset = create_dataset(train_dir, args.batch_size, True)
    val_dataset = create_dataset(val_dir, args.batch_size)
    
    weights = np.load(os.path.join(args.data_dir, 'class_weights.npy'))
    class_weights = dict(zip([0,1,2,3], weights))
    
    # Create tensorflow model and train/validate
    model = model(train_dataset, val_dataset, class_weights, args.epochs)
    
    # Save model
    model.save(os.path.join(args.sm_model_dir, '000000001'))

