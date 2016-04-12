# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:35:25 2016

@author: songyayong
"""

from math import log
#一个数据集的熵=求和{每一个分类的概率p(x)*这个分类的信息}，分类x的信息l(x)=-log(p(x),2)
def calShannonEnt(dataSet):
    #假设，数据最后一列是分类标签，每一行是一条记录
    numEntries = len(dataSet)
    labelCounts = {}
    #创建字典，统计每个分类数量    
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #求信息熵，对每个分类（概率*信息）求和
    for key in labelCounts:
        #每个分类概率        
        prob = float(labelCounts[key])/numEntries
        #每个分类的概率*信息        
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

#输入数据集，特征轴，特征值，提取在特征轴上与特征值一致的元素，形成一个数据子集，子集中的元素没有特征轴对应的特征值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #python列表拼接
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
def chooseBestFeatureToSplit(dataSet):
    #特征个数，最后一列是分类    
    numFeatures = len(dataSet[0]) -1
    #先计算原始数据集的熵
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        #提取所有元素一个特征值
        featList = [example[i] for example in dataSet]
        #放到set中，唯一化        
        uniqueVals = set(featList)
        newEntropy = 0.0
        #迭代该特征中的每一个值，计算该类别的熵
        for value in uniqueVals:
            #划分出所有特征i，值为value的子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calShannonEnt(subDataSet)
        #比较按特征i分类后的熵和原始熵的差别
        infoGain = baseEntropy - newEntropy
        #熵减少得最大，说明分类效果最好，因为信息都在这个特征上        
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature