# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:23:30 2016

@author: songyayong
"""
from numpy import *
#numpy本身就是log，是支持矢量操作的，python本身的math是不支持的
#from math import log

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #文档手工分类，用于训练，有监督，分别对应上面6条文档（分词结果），1表示侮辱性文字
    return postingList,classVec
#把原始数据的字条都加入到一个list中
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        #两个set合并
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
#两个参数，词汇表（总向量，就是将全部文档的词汇合到一起），输入的文档词汇，输出的词汇向量是词集模型，是看单词是否出现，不论次数
def setOfWords2Vec(vocabList,inputSet):
    #先生成一个在所有维度都是0的向量 list*num就是扩大num倍,向量位置和在字典list位置是一一对应的
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        #x in list , set 都行
        if word in vocabList:
            #每个单词，只有1，0两处状态            
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" %word
    return returnVec
#两个参数，词汇表（总向量，就是将全部文档的词汇合到一起），输入的文档词汇，输出的词汇向量是词袋模型，计算单词出现次数
def bagOfWords2VecMN(vocabList,inputSet):
    #先生成一个在所有维度都是0的向量 list*num就是扩大num倍,向量位置和在字典list位置是一一对应的
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        #x in list , set 都行
        if word in vocabList:
            #这里计算的是词频
            returnVec[vocabList.index(word)] +=1
        else: print "the word: %s is not in my Vocabulary!" %word
    return returnVec
def trainNB0(trainMatrix,trainCategory):
    #文档个数，统计类别概率时的分母   
    numTrainDocs = len(trainMatrix)
    #词频维度，也就是特征维度，同时也是独立变量个数    
    numWords = len(trainMatrix[0])
    #这里因为文档标签是0和1，所有对标签求和，就得到分类是1的文档个数，这是个例。
    #这里得到分类1，也就是侮辱性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
#    p0Num = zeros(numWords);
#    p1Num = zeros(numWords);
#    p0Denom = 0.0;
#    p1Denom = 0.0;    
    #数据稀疏有0，乘积会变成0，初始化就从1和2开始    
    p0Num = ones(numWords);
    p1Num = ones(numWords);
    p0Denom = 2.0;
    p1Denom = 2.0;    

    
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            #在每个词向量上累加            
            p1Num += trainMatrix[i]
            #单纯统计词汇量，只不过是在这个分类下            
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #得到每个词的出现概率向量，在类型1中，原始计算值可以太小了，连续相乘就变得太小而溢出了，用取自然对数方式最好，两者会同增同减，同时取极值。
    #最后的分类结果只判断相对大小，因此只要变化趋势是一致的就可以
#    p1Vect = p1Num/p1Denom
    p1Vect = log(p1Num/p1Denom)
    #得到每个词的出现概率，在类型0中    
#    p0Vect = p0Num/p0Denom
    p0Vect = log(p0Num/p0Denom)
    #返回每个单词的条件概率向量，和每个分类的概率    
    return p0Vect,p1Vect,pAbusive
#输入是样本向量，每个分类条件下的单词概率向量，目标分类的概率(这里正好两种分类，所有另一种分类概率就是1-p了)
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #这作者真是简化过了，因为这些概率结果是log过的，log相加就相当于是概率计算时相乘了！
    #这里就可以看出来log计算的好处，把原本多个条件概率相乘的计算，变成了多个概率相加了！p(w|c)*    
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    #原本这里的p1,p2还要除以一个p(w)，即向量本身的概率，但是最好两者还要比较，两边可以都省略掉这一步！    
    if p1 > p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClass = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClass))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,' classified as :',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,' classified as :',classifyNB(thisDoc,p0V,p1V,pAb)

def textParse(bigString):    #input is big string, #output is word list
    import re
    #正则匹配，只取单词数字    
    listOfTokens = re.split(r'\W*', bigString)
    #过滤长度小于2的单词，并且全部小写化    
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] 
    
def spamTest():
    #分词后文档列表，分类列表，单词列表
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    #一共50个文档    
    trainingSet = range(50); testSet=[]           #create test set
    #从中抽取10个作为测试集，剩下40个，这里抽取的方式是只对索引进行增删，实际存储没变    
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    #用索引来提取数据，比转移数据什么的方便多了
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText