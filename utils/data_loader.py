from pandas import json

import numpy as np
import os
import glob

import tensorflow as tf

# path to the images and the text file which holds the scores and ids
base_images_path = r'/home/cola/work/nenet/nima/images-data/AVA_dataset/images/images/'
ava_dataset_path = r'/home/cola/work/nenet/nima/images-data/AVA_dataset/AVA.txt'

tip_base_images_path = r'/home/cola/work/nenet/nima/images-data/tid2013/distorted_images/'
tip_dataset_path = r'/home/cola/work/nenet/nima/images-data/tid2013/TID2013/tid_labels_train.json'


IMAGE_SIZE = 224

print 'base_images_path = ' + base_images_path
files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

files = glob.glob(base_images_path + "*.bmp")
files = sorted(files)
g_train_image_paths = []
g_train_scores = []
g_val_image_paths = []
g_val_scores = []


def load_tid_data():
    global g_train_image_paths
    global g_train_scores
    global g_val_image_paths
    global g_val_scores
    print("Loading training set and val set")
    f = open(tip_dataset_path, 'r')
    data = json.load(f)
    image_size = 0
    for item in data:
        image_id = item['image_id']
        label = item['label']
        file_path = tip_base_images_path + image_id + '.bmp'
        if os.path.exists(file_path):
            g_train_image_paths.append(file_path)
            g_train_scores.append(label)
            image_size = image_size + 1
    g_train_image_paths = np.array(g_train_image_paths)
    g_train_scores = np.array(g_train_scores, dtype='float32')

    g_val_image_paths = g_train_image_paths[-5000:]
    g_val_scores = g_train_scores[-5000:]
    train_image_paths = g_train_image_paths[:-5000]
    train_scores = g_train_scores[:-5000]
    print('Train set size : ', train_image_paths.shape, train_scores.shape)
    print('Val set size : ', g_val_image_paths.shape, g_val_scores.shape)
    print('Train and validation datasets ready !')


def load_ava_data():
    global g_train_image_paths
    global g_train_scores
    global g_val_image_paths
    global g_val_scores
    print("Loading training set and val set")
    f = open(ava_dataset_path, 'r')
    lines = f.readlines()
    image_size = 0
    for i, line in enumerate(lines):
        token = line.split()
        image_id = int(token[1])

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = base_images_path + str(image_id) + '.jpg'
        if os.path.exists(file_path):
            g_train_image_paths.append(file_path)
            g_train_scores.append(values)
            image_size = image_size + 1

        if image_size >= 10000:
            break

        count = 255000 // 20
        if i % count == 0 and i != 0:
            print('Loaded %d percent of the dataset' % (i / 255000. * 100))

    g_train_image_paths = np.array(g_train_image_paths)
    g_train_scores = np.array(g_train_scores, dtype='float32')

    g_val_image_paths = g_train_image_paths[-5000:]
    g_val_scores = g_train_scores[-5000:]
    train_image_paths = g_train_image_paths[:-5000]
    train_scores = g_train_scores[:-5000]
    print('Train set size : ', train_image_paths.shape, train_scores.shape)
    print('Val set size : ', g_val_image_paths.shape, g_val_scores.shape)
    print('Train and validation datasets ready !')




def parse_data(filename, scores):
    '''
    Loads the image file, and randomly applies crops and flips to each image.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (256, 256))
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def train_generator(batchsize, shuffle=True):
    '''
    Creates a python generator that loads the AVA dataset images with random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((g_train_image_paths, g_train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)

def val_generator(batchsize):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for validation set

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset.from_tensor_slices((g_val_image_paths, g_val_scores))
        val_dataset = val_dataset.map(parse_data_without_augmentation)

        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)

def features_generator(record_path, faeture_size, batchsize, shuffle=True):
    '''
    Creates a python generator that loads pre-extracted features from a model
    and serves it to Keras for pre-training.

    Args:
        record_path: path to the TF Record file
        faeture_size: the number of features in each record. Depends on the base model.
        batchsize: batchsize for training
        shuffle: whether to shuffle the records

    Returns:
        a batch of samples (X_features, y_scores)
    '''
    with tf.Session() as sess:
        # maps record examples to numpy arrays

        def parse_single_record(serialized_example):
            # parse a single record
            example = tf.parse_single_example(
                serialized_example,
                features={
                    'features': tf.FixedLenFeature([faeture_size], tf.float32),
                    'scores': tf.FixedLenFeature([10], tf.float32),
                })

            features = example['features']
            scores = example['scores']
            return features, scores

        # Loads the TF dataset
        train_dataset = tf.data.TFRecordDataset([record_path])
        train_dataset = train_dataset.map(parse_single_record, num_parallel_calls=4)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=5)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        # indefinitely extract batches
        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)