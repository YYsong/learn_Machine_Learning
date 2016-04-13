# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:35:25 2016

@author: songyayong
"""

from math import log
import operator
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
#找到一个列表内出现次数最多的元素
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]==0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#输入是数据集，和特征标签列表，也就是第几列叫什么
def createTree(dataSet,labels):
    #生成标签列表（从最后一列得到）    
    classList = [example[-1] for example in dataSet]
    #递归判断1：当前的特征选择已经使得标签类型都一样了，返回-----已经分好类了    
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #递归判断2：当前已经没有特征了，只有最后一列标签-----已经提取完了所有特征，此时仍然没有把不同类标签分离开，只能返回数量最多的标签作为分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #开始对数据集进行划分，得到最优特征，实际上就是哪一列
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #得到这个特征的标签    
    bestFeatLabel = labels[bestFeat]
    #声明一个字典，存储分类树，分类标签，值也是字典    
    myTree = {bestFeatLabel:{}}
    #从标签列表中删除已选择的特征，这里用的是python的内置方法del(array[index])，是直接作用于列表的    
    del(labels[bestFeat])
    #获取特征列的所有属性
    featValues = [example[bestFeat] for example in dataSet]
    # 唯一化，用python set最快
    uniqueVals = set(featValues)
    #每一个特征值都会生成一个子树
    for value in uniqueVals:
        #子树的特征标签是删除已经选择特征值的标签，这里要再复制一遍，是因为subLabels一会儿要在递归里用，引用传递会出错
        subLabels = labels[:]
        #递归调用，每一个特征值，都要再生成一个子分类树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree