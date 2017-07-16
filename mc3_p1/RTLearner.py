"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import random
import math

class RTLearner(object):

    def __init__(self,leaf_size=1, verbose=False):
        self.lfSize=leaf_size

    def addEvidence(self,dataX,dataY):
        
        myData=np.column_stack((dataX,dataY))

        self.RT=self.buildRT(self.lfSize,myData)
    def buildRT(self,lf_sz,data):
        if data.shape[0]<=lf_sz:

            return {'name':'leaf','feature':np.nan, 'val':np.mean(data[:,-1]),\
            'left':np.nan,'right':np.nan}
        elif np.all(data[:,-1]==data[0,-1]):
            return {'name':'leaf','feature':np.nan, 'val':data[0,-1],\
            'left':np.nan,'right':np.nan}
        else:
            idx=random.randint(0,data.shape[1]-2)
            idx1=random.randint(0,data.shape[0]-1)
            idx2=random.randint(0,data.shape[0]-1)
            
            

            splitValue=(data[idx1,idx]+data[idx2,idx])/2
            if data[data[:,idx]>splitValue].shape[0]==0:
                return self.buildRT(lf_sz,data)
            else:
            
               tree={'name':'node','feature':idx,'val':splitValue,\
                'left':self.buildRT(lf_sz,data[data[:,idx]<=splitValue]),\
                'right':self.buildRT(lf_sz,data[data[:,idx]>splitValue])}
        return tree

    def query(self,points):
        y=[]
        for x in points:
            t=self.RT
            while t['name']=='node':
                if x[t['feature']]<=t['val']:
                    t=t['left']
                else:
                    t=t['right']
            y.append(t['val'])
        return np.array(y)

        
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
