#coding:utf-8
from gplearn.genetic import SymbolicTransformer
from sklearn.utils import check_random_state
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import math
import xlrd
import os

fname1='C:\\Users\\Lenovo\\Desktop\\data\\'
all_data = np.loadtxt(fname1+'out0805_3.txt')
print all_data.shape
m = 7460
n = 7510
data = all_data[m:n,:]
data[:,[2]] = (data[:,[7]]-data[:,[2]])
data[:,[12]] = (data[:,[7]]-data[:,[12]])
features = data[:,[4,12,13,14,15]]
labels1 = data[:,[6]]
x = int(0.7*(n-m))
data_train = features[0:x]
data1_train = labels1[0:x]
data_test = features[x:]
data1_test = labels1[x:]

est_gp = SymbolicRegressor(population_size=20000, init_depth=(3,15),
                           generations=20, stopping_criteria=0.01,
                           comparison=True, transformer=True,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,const_range=(-20.0, 20.0),
                           parsimony_coefficient=0.01, random_state=1, metric='mse')
						   
est_gp.fit(data_train,data1_train)
print est_gp._program
score_gp = est_gp.score(data_test, data1_test)
print  score_gp
p=est_gp.predict(features)

xc=np.arange(0,len(labels1),1)
xa=np.arange(0,x,1)
xb=np.arange(x-1,len(labels1),1)
#print xb.size
if xb.size!=len(p[x-1:]):
   xb=np.arange(x-1,len(labels1)-1,1)


#print xa.size, xb.size,xc.size
print len(p[0:x]),len(p[x-1:]),len(labels1)
print p,labels1

fig = plt.figure()
fig.patch.set_facecolor('white')
line1, = plt.plot(xa,p[0:x],'-b')
line2,=plt.plot(xb,p[x-1:],'--r')
line3,=plt.plot(xc,labels1,'-g')
plt.legend((line1, line2,line3), ("Predicted Acceleration", 'Simulated Acceleration','Actual Acceleration',),loc=1)
plt.xlabel("Time(0.1s)")
plt.ylabel("Acceleration(feet/s^2)")
plt.show()

fname = fname1+'newstraight_result\\out0805_3.'+str(m)+'.'+str(n)
fig.savefig(fname+'.png',format='png')
f = open(fname+'.txt','w')
f.writelines([str(i)+'\n' for i in p])
f.close()