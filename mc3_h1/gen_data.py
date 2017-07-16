import numpy as np
import random
import math

def best4LinReg():
	
	rows = random.randint(20, 100)
	cols = random.randint(5,100)
	alpha=np.random.randint(-3,8,cols)
	X=15*np.random.random_sample([rows,cols])
	Y=np.array(list(np.dot(X[i,:],(alpha +np.random.random_sample(cols))) for i in range(rows)))
	
	return X,Y

def best4RT():
	
	rows = random.randint(20, 200)
	cols = random.randint(5,15)
	X=np.array(list(np.random.randint(1,100)*np.random.random_sample(cols) for i in range(rows)))
	Y= np.array(list(np.inner(X[i,:],X[i,:])*np.random.random_sample() for i in range(rows)))

	return np.concatenate((X,X,X,X),axis=0),np.concatenate((Y,Y,Y,Y),axis=0)

