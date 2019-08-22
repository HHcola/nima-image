# -*- coding: utf-8 -*
import numpy
import scipy.stats as stats

# Linear Correlation Coefficient, LCC 皮尔森相关系数
from utils.score_utils import mean_score


def lcc(y_true, y_pred):
    return numpy.corrcoef(y_true, y_pred)[0, 1]


# Spearman's Rank  Correlation Coefficient, SRCC 斯皮尔曼相关性系数
def srcc(y_true, y_pred):
    return stats.stats.spearmanr(y_true, y_pred)


