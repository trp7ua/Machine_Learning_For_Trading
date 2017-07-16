"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import random
import math
import RTLearner as rt

class BagLearner(object):

    def __init__(self,learner = rt.RTLearner, 
         bags = 1, 
        boost = False, verbose = False, **kwargs):
        self.bg=bags
        self.lfSize=kwargs['kwargs']['leaf_size']
        self.learners=[]
        for i in range(0,bags):
            self.learners.append(learner(leaf_size=self.lfSize,verbose=False))

    def addEvidence(self,dataX,dataY):

        for i in range(0,self.bg):
            myData=np.column_stack((dataX,dataY))
            bootstrapSample=myData[np.random.choice(np.arange(len(myData)),\
                len(myData))]
            self.learners[i].addEvidence(bootstrapSample[:,:-1],\
                bootstrapSample[:,-1])

    def query(self,points):
        y=[]
        for i in range(0,self.bg):
            y.append(self.learners[i].query(points))
        z=np.vstack(y)
        return np.mean(z,axis=0)
        

        
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
