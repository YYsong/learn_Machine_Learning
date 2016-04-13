# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:41:37 2016

@author: songyayong
"""
import kNN
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
kNN.datingClassTest()
kNN.classifyPerson()
kNN.handwritingClassTest()

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()

