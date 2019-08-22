# -*- coding: utf-8 -*
import argparse
import os

from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot
import tensorflow as tf

from utils.data_loader import train_generator, val_generator, load_tid_data, load_ava_data
from utils.evaluation import srcc, lcc, tf_pearson
from utils.threadsafe import threadsafe_generator

'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, val_gen=0, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.val_gen = val_gen
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)
        # numpy.corrcoef(list1, list2)[0, 1]  from scipy.stats import spearmanr
        self.writer.flush()

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


def earth_mover_loss_tanh(y_true, y_pred):
    cdf_ytrue = K.cumsum(K.tanh(y_true), axis=-1)
    cdf_ypred = K.cumsum(K.tanh(y_pred), axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

parser = argparse.ArgumentParser(description='Data Load')
parser.add_argument('-type', type=str, default='ava',
                    help='Pass a data type to train')

parser.add_argument('-weight', type=str, default='merge',
                    help='Weight:merge(weight),ava_weight,tid_weight')

args = parser.parse_args()

AVA_DATA_TYPE = 'ava'
TID2013_DATA_TYPE = 'tid'

data_type = 'ava'
if args.type is not None:
    data_type = args.type

WEIGHT_TYPE_MERGE = 'merge'
WEIGHT_TYPE_AVA = 'ava_weight'
WEIGHT_TYPE_TID = 'tid_weight'

weight_type = 'merge'
if args.weight is not None:
    weight_type = args.weight

if data_type == TID2013_DATA_TYPE:
    load_tid_data()
    image_all_size = 3000
else:
    load_ava_data()
    image_all_size = 63461


image_size = 224

base_model = MobileNetV2((image_size, image_size, 3), alpha=1.0, include_top=False, pooling='avg')
for layer in base_model.layers:
    layer.trainable = False

# Dropout 正则化，防止过拟合
# rate：0-1
x = Dropout(0.75)(base_model.output)

# Dense 全连接层
# units：正整数，输出空间维度
# activation：激活函数
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.summary()


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def lccv(y_true, y_pred):
    return tf_pearson(y_true, y_pred)
    # return tf.convert_to_tensor(r_lcc)


# 优化器
optimizer = Adam(lr=1e-3)
model.compile(optimizer, loss=earth_mover_loss_tanh, metrics=[lccv, srcc])

# tensorflow variables need to be initialized before calling model.fit()
# there is also a tf.global_variables_initializer(); that one doesn't seem to do the trick
# You will still get a FailedPreconditionError
K.get_session().run(tf.local_variables_initializer())

if weight_type == WEIGHT_TYPE_MERGE:
    model_weights_path = 'weights/mobilenet_v2_weights.h5'
elif weight_type == WEIGHT_TYPE_AVA:
    model_weights_path = 'weights/mobilenet_v2_ava_weights.h5'
elif weight_type == WEIGHT_TYPE_TID:
    model_weights_path = 'weights/mobilenet_v2_tid_weights.h5'
else:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                model_weights_path = 'weights/mobilenet_v2_weights.h5'

print ''
# load weights from trained model if it exists
if os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)

checkpoint = ModelCheckpoint(model_weights_path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,
                             mode='min')


batchsize = 200
epochs = 5
steps = 1

val_gen = threadsafe_generator(val_generator(batchsize=batchsize))
tensorboard = TensorBoardBatch(val_gen)
callbacks = [checkpoint, tensorboard]


# steps_per_epoch 一个epoch包含的步数（每一步是一个batch的数据输入） steps_per_epoch = image_size(63461) / batchsize
# validation_steps=ceil(val_dataset_size/batch_size),
history = model.fit_generator(train_generator(batchsize=batchsize),
                    steps_per_epoch=steps,
                    epochs=epochs, verbose=1, callbacks=callbacks,
                    validation_data=val_gen,
                    validation_steps=20)

# plot metrics
pyplot.plot(history.history['lccv'])
pyplot.show()

pyplot.plot(history.history['srcc'])
pyplot.show()

