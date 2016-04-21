# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 08:39:58 2016

@author: songyayong
"""

from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #初始化返回结果，默认是1    
    retArray = ones((shape(dataMatrix)[0],1))
    #阈值判断方式是小于    
    if threshIneq == 'lt':
        #小于判断，则把小于阈值数值都变成-1
        #这里的用法比较独特，矩阵做判断得到一个同型判断矩阵，一个矩阵嵌套一个同型判断矩阵，是做了一个筛选功能
        #如果筛选后面再加一个赋值，就变成了对符合条件（结果是true）的对应元素赋值，是原地处理的
        retArray[dataMatrix[:,dimen]<=threshVal] = -1.0
    else :
        #大于判断，则把大于阈值数值都变成-1
        retArray[dataMatrix[:,dimen]>threshVal] = -1.0
    return retArray

#输入数据集，标签和数据的权重向量
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    #无限大常量
    minError = inf
    #对每一列特征做迭代
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        #最大-最小/迭代次数，得到步长
        stepsize = (rangeMax-rangeMin)/numSteps
        #对每个步长对循环
        for j in range(-1,int(numSteps)+1):
            #对每一个不等号循环
            for inequal in ['lt','gt']:
                #阈值逐渐增加
                threshVal = (rangeMin+float(j)*stepsize)
                #根据阈值分类了
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                #计算错的，就是1，正确的是0，还是矩阵过滤赋值大法！判断可以是同型对应对比的
                errArr[predictedVals == labelMat]=0
                #错误数量和权重相乘，得到加权后的错误数
                weightedError = D.T*errArr
#                print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                #更新最小错误
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    #得到某个维度上的最佳阈值，声明判断方法 大于是1，小于是-1
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#输入数据集，标签，迭代次数    
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    #存储弱分类器的列表
    weakClassArr=[]
    m = shape(dataArr)[0]
    #初始样本权重都一样，样本向量权重之和是1，一开始都1/m
    D = mat(ones((m,1))/m)
    #每一个数据类别的估计值
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print "D:",D.T
        #计算当前分类器的权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        #分类器列表添加当前分类器
        weakClassArr.append(bestStump)
        print "classEst: ",classEst.T
        #为下一轮迭代计算D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate,"\n"
        #当错误率为0时退出        
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    #初始估计都是0
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        #叠加每一分类结果*权重        
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)
    
    