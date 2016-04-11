# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#参数：分类的输入向量，训练样本集，标签向量，最近个数k,其中矩阵行数和样本标签数目一致
def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    #计算欧式距离
    #一个4行1列的矩阵（输入集*行数）-4*1矩阵（训练集），临到每个相差    
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #相差平方
    sqDiffMat = diffMat**2
    #平方和    
    sqDistances = sqDiffMat.sum(axis=1)
    #开方    
    distances = sqDistances**0.5
    
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]