from pandas import json

import numpy as np
import os
import glob

import tensorflow as tf

'''
Checks all images from the AVA dataset if they have corrupted jpegs, and lists them for removal.

Removal must be done manually !
'''

base_images_path = r'/home/cola/work/nenet/nima/images-data/tid2013/distorted_images/'
tid_dataset_path = r'/home/cola/work/nenet/nima/images-data/tid2013/TID2013/tid_labels_train.json'


IMAGE_SIZE = 128
BASE_LEN = len(base_images_path) - 1

files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
with open(tid_dataset_path, mode='r') as f:
    f = open(tid_dataset_path, 'r')
    data = json.load(f)
    for item in data:
        image_id = item['image_id']
        label = item['label']
        file_path = base_images_path + image_id + '.bmp'
        file_path_BMP = base_images_path + image_id + '.BMP'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(label)
        elif os.path.exists(file_path_BMP):
            train_image_paths.append(file_path_BMP)
            train_scores.append(label)


train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

def parse_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_image(image, channels=3)
    return image

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())

    count = 0
    fn = tf.placeholder(dtype=tf.string)
    img = parse_data(fn)

    for path in train_image_paths:
        try:
            sess.run(img, feed_dict={fn: path})
        except Exception as e:
            # rename file
            os.rename(path, path + '_bk')
            print(path, "failed to load !" + e.message)
            print()
            count += 1

        # try:
        #     img = Image.open(path)
        # except IOError:
        #     # rename file
        #     os.rename(path, path + '_bk')
        #     print(path, "failed to load !" + e.message)
        #     print()
        #     count += 1
        #
        # # noinspection PyBroadException
        # try:
        #     img = np.asarray(img)
        # except:
        #     print('corrupt img', path)

    print(count, "images failed to load !")

print("All done !")

"""
Had to delete file : 440774.jpg and remove row from AVA.txt
Had to delete file : 179118.jpg and remove row from AVA.txt
Had to delete file : 371434.jpg and remove row from AVA.txt
Had to delete file : 277832.jpg and remove row from AVA.txt
Had to delete file : 230701.jpg and remove row from AVA.txt
Had to delete file : 729377.jpg and remove row from AVA.txt
"""