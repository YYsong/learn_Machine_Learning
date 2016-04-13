# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#参数：分类的输入向量，训练样本集，标签向量，最近个数k,其中矩阵行数和样本标签数目一致
def classify0(inX, dataSet, labels, k):
    #numpy矩阵的shape=(行数，列数)元组    
    dataSetSize = dataSet.shape[0]

    #计算欧式距离
    #一个4行1列的矩阵（输入集*行数）-4*1矩阵（训练集），临到每个相差    
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
#    print diffMat
    #相差平方
    sqDiffMat = diffMat**2
#    print sqDiffMat
    #在第一列上求和，变成一个numpy数组    
    sqDistances = sqDiffMat.sum(axis=1)
#    print sqDistances
    #开方    
    distances = sqDistances**0.5
#    print type(distances)
    #numpy数据的排序下标，默认从小到大排，正好是距离最近的优先
    sortedDistIndicies = distances.argsort()
#    print sortedDistIndicies
    #标签字典，key是标签，value是出现次数    
    classCount={}
    #在前k名，也就是前k个最近的元素
    for i in range(k):
        #得到标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #在标签字典中统计
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #把标签字典按照value，也就是元组的第二个元素进行逆序排序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#读取文件到矩阵中
def file2matrix(filename):
    fr= open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #建立矩阵，n行，3列（3个特征值）
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        #strip可以去掉字符串开头结尾多余字符        
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        #提取int标签
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

#数值归一化
def autoNorm(dataSet):
    #参数0表示按列取矩阵的最小值，1是按行
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    #生成一个与目标矩阵形状一样的0矩阵
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #原矩阵-一个由最小值组成的复制出来的同形矩阵
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals
#测试分类器
def datingClassTest():
    #测试数据占总数据的比例，这里原始数据原本就是随机的，因此直接随意取10%的数据就行    
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges, minVals = autoNorm(datingDataMat)
    #记录总数    
    m = normMat.shape[0]
    #抽样总数
    numTestVecs = int(m*hoRatio)
    #数值变成float型，做除法后才是float型    
    errorCount = 0.0
    #第1行到第numTesetVecs，依次与其余90%数据进行kNN分类，k值取3，得到分类标签，然后再与真实的标签做比较
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifer came back with: %d, the real answer is : %d" %(classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount +=1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
#测试一个人的分类（外部数据记录分类）
def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of icecream consumed per year?"))
    #生成训练集    
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    #生成归一化数据    
    normMat,ranges, minVals =autoNorm(datingDataMat)
    #生成输入    
    inArr = array([ffMiles,percentTats,iceCream])
    #将归一化后的输入数据和训练集，标签和k值放入分类器    
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]
#将32*32的bit字符转化为一个1024*1的行向量
def img2vector (filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    #建立训练集    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        #取文件名的关键字作为标签        
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr)
        if (classifierResult != classNumStr) : errorCount += 1.0
    print "\nthe total number of errors is : %d" % errorCount
    print "\nthe total error rate is : %f" % (errorCount/float(mTest))