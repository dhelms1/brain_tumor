import tensorflow as tf
import numpy as np
import argparse
import os
import json
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

def model(train_dataset, val_dataset, class_weights, epochs):
    '''
    Create a TensorFlow model and train/validate on the given data. Returns
    the final model.
    '''
    # CREATE MODEL  
    
    return model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for dataset (default=32)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='input epochs for training (default=15)')
    parser.add_argument('--class_weights', type=float, default=None, metavar='N',
                        help='class weights for training (default=None)')
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    args = parser.parse_args()
    
    # Create training and validation datasets
    train_dir = os.path.join(args.data_dir, 'train_images.tfrecords')
    val_dir = os.path.join(args.data_dir, 'val_images.tfrecords')
    
    train_dataset = create_dataset(train_dir, args.batch_size)
    val_dataset = create_dataset(val_dir, args.batch_size)
    
    # Create tensorflow model and train/validate
    model = model(train_dataset, val_dataset, args.class_weights, args.epochs)
    
    # Save model
    model_path = os.path.join(args.model_dir, 'my_model')
    model.save(model_path)

