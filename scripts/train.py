import tensorflow as tf
import numpy as np
import argparse
import os
AUTOTUNE = tf.data.AUTOTUNE

def tfrecord_reader(record):
    '''
    Parse a single tfrecord and return the image (formatted as a tensor)
    and the label (formatted as an integer).
    '''
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed = tf.parse_single_example(record, features)
    return ({'data': tf.io.parse_tensor(parsed['image_raw'], tf.float32)}, 
            parsed['label'].numpy())

def load_data(data_dir):
    '''
    Create a TFRecordDataset and decode a given tfrecord file.
    '''
    data = tf.data.TFRecordDataset(data_dir)
    data = data.map(tfrecord_reader)
    return data

def create_dataset(data_dir, BATCH_SIZE):
    '''
    Create and parse the tfrecord file for a given dataset. Turn dataset
    into batches of specified size.
    '''
    dataset = load_data(data_dir)
    dataset = dataset.shuffle(1000)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def model(X_train, y_train, X_val, y_val):
    '''
    Create a TensorFlow model and train/validate on the given data.
    '''
    # IMPORT MODEL (or build)  
    
    return model

if __name__ == "__main__":
    
    # add parser
    
    X_train, y_train = create_dataset(train_dir, BATCH_SIZE)
    X_val, y_val = create_dataset(val_dir, BATCH_SIZE)
    
    model = model(X_train, y_train, X_val, y_val)

