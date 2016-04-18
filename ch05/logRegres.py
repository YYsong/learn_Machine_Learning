# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:11:52 2016

@author: songyayong
"""
from numpy import *
#导入数据源，数据格式是x1,x2,label
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #数据行向量是常数项，变量1，变量2
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        #标签向量        
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#Sigmoid函数，把一个实数放到0~1之间
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    #重复150次迭代样本数据集，这次系数能收敛在一个稳定范围
    for j in range(numIter): 
        dataIndex = range(m)
        for i in range(m):
            #每次迭代时需要调整alpha,其实是alpha一开始大一点，迭代剧烈些，后面变小，收敛好
            alpha = 4/(1.0+j+i)+0.01
            #随机选取一个样本
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights + alpha*error*dataMatrix[randIndex]
            #原地删除list中的一个元素
            del(dataIndex[randIndex])
    return weights

#随机/在线/增量 梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    #获取样本规模    
    m,n = shape(dataMatrix)
    alpha = 0.01
    #初始化系数，这里是一个array，而不是一列！    
    weights = ones(n)
    #迭代次数是数据量，比较小
    for i in range(m):
        #这里是每行数据做一次计算，得到差值都是数值而不是向量！
        h = sigmoid(sum(dataMatrix[i]*weights))
        #仅仅根据这一条记录，计算这次的误差
        error = classLabels[i] - h
        #仅仅根据这一次误差，修改一回系数
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    #把标签列表矩阵转置！从n*1变成1*n
    labelMat = mat(classLabels).transpose()
    #得到输入矩阵的行数和列数
    m,n = shape(dataMatrix)
    #步长    
    alpha = 0.001
    #最大迭代次数    
    maxCycles = 500
    #系统矩阵，行数要与输入矩阵的列数（变量数）一致，初始化为1（变量不变）    
    weights = ones((n,1))
    #迭代多次，求出系数矩阵
    for k in range(maxCycles):
        #m行n列的矩阵，乘以n行1列系数矩阵，变成m行1列的矩阵，矩阵再用sigmoid函数处理，得到范围是0~1之间的数值，这是结果矩阵
        h = sigmoid(dataMatrix*weights)
        #标签矩阵（真实分类结果，用0或者1来表示）-结果矩阵（数值都在0~1之间），得到差值矩阵m行1列，代表真实类别和预测类别的差值，这个差值是由真实-预测得到的，是增量结果
        error = (labelMat - h)
        #数据集转置变成n行m列，与差值矩阵m行1行相乘，变为n行1列的差值系数矩阵，代表了梯度向量，再与alpah系数相乘得到梯度，与原来系数相加得到更新后的系数
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

#输入是回归得到的系数
#执行时要用 logRegres.plotBestFit(Weights.getA())，把weights从矩阵变为array
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [] ; ycord1 = []
    xcord2 = [] ; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    #从-3到3,生成60个数，间隔0.1，这些是分隔线的横坐标
    x = arange(-3.0,3.0,0.1)
    #分隔线的纵坐标x2应该满足w0x0+w1x1+w2x2=0,0输入到sigmoid函数正好是0.5，就是两个分类的分隔线
    #这里画直线没有用方程，而是以密集的点来代表面，因为有时候分隔线可以不是直线，这样的画法更规范
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()