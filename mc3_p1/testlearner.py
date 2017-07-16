"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import RTLearner as rt
import BagLearner as bl
import sys
import random

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    
    # create a learner and train it
    learner = bl.BagLearner(learner = rt.RTLearner, 
        kwargs = {"leaf_size":1}, bags = 20, 
        boost = False, verbose = False) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it

    predY = learner.query(testX) # get the predictions
    
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
