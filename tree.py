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