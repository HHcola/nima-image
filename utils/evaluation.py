# -*- coding: utf-8 -*
import numpy
import scipy.stats as stats
from keras import backend as K
import tensorflow as tf

# Linear Correlation Coefficient, LCC 皮尔森相关系数
from utils.score_utils import mean_score


def lcc(y_true, y_pred):
    return numpy.corrcoef(y_true.numpy(), y_pred.numpy(), rowvar=False)[0, 1]


# Spearman's Rank  Correlation Coefficient, SRCC 斯皮尔曼相关性系数
def srcc(y_true, y_pred):
    return tf.py_function(stats.stats.spearmanr,
                          [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout=tf.float32)
    # return stats.stats.spearmanr(y_true, y_pred, axis=None)


def tf_pearson(y_true, y_pred):
    return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]


