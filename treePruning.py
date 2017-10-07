import numpy as np
import treeBuilding as tb
import piecewiseLinear as pl

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = tb.binSplitDataSet(testData, tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = tb.binSplitDataSet(testData, tree['spInd'],tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) + sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print ("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

myMat2 = np.mat(tb.loadDataSet('exp2.txt'))
print(tb.createTree(myMat2, pl.modelLeaf, pl.modelErr,(1,10)))
