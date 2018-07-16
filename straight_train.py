#coding:utf-8
from gplearn.genetic import SymbolicTransformer
from sklearn.utils import check_random_state
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
import math
import xlrd
import copy
import mahotas as mh
fname1="C:\Users\Lenovo\Desktop\data\carcarcar\data\\"
fname2='01.39.2'
fname=fname1+fname2+'.xlsx'
bk=xlrd.open_workbook(fname)
sh = bk.sheet_by_name("Sheet1")
m=sh.nrows
n=sh.ncols
#print m ,n
row_list = []
#获取各行数据
for i in range(0,m):
    row_data = sh.row_values(i)
    row=[]
    for j in range(0,n):
        row.append(float(row_data[j]))
    row_list.append(row)

data=np.array(row_list)
#data=mh.gaussian_filter(data,1)
#print data
da=data[:,[2,4,5,6]]
tar=data[:,[1]]
x=int(0.85*m)
#x=m-1
da_train=da[0:x]
tar_train=tar[0:x]
da_test=da[x:]
tar_test=tar[x:]

#遗传算法
est_gp = SymbolicRegressor(population_size=30000, init_depth=(2,10),
                           generations=20, stopping_criteria=0.01,
                           comparison=True, transformer=True,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,const_range=(-20.0, 20.0),
                           parsimony_coefficient=0.01, random_state=1)

est_gp.fit(da_train, tar_train)

print est_gp._program
score_gp = est_gp.score(da_test, tar_test)
print  score_gp
p=est_gp.predict(da)
print r2_score(tar,p)
print explained_variance_score(tar,p)
print mean_squared_error(tar,p)

 #决策树
est_dt = DecisionTreeRegressor()
est_dt = est_dt.fit(da_train,tar_train)
p1 = est_dt.predict(da)
print est_dt.score(da_test,tar_test)
print r2_score(tar,p1)
print explained_variance_score(tar,p1)
print mean_squared_error(tar[x:],p1[x:])


#随机森林
est_rfc = RandomForestRegressor()
est_rfc = est_rfc.fit(da_train,tar_train)
p2 = est_rfc.predict(da)
print est_rfc.score(da_test,tar_test)
print r2_score(tar,p2)
print explained_variance_score(tar,p2)
print mean_squared_error(tar[x:],p2[x:])

#贝叶斯
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
est_gbr = ensemble.GradientBoostingRegressor(**params)
est_gbr.fit(da_train, tar_train)
p3 = est_gbr.predict(da)
print est_gbr.score(da_test,tar_test)
print r2_score(tar,p3)
print explained_variance_score(tar,p3)
print mean_squared_error(tar[x:],p3[x:])


#print tar,p

xc=np.arange(0,len(tar), 1)
xa=np.arange(0,len(da_train),1)
xb=np.arange(len(da_train)-1,len(tar),1)
fig = plt.figure()
fig.patch.set_facecolor('white')
line1,= plt.plot(xc[0:x],p[0:x],'-b')
line2,= plt.plot(xc[x-1:],p[x-1:],'-r')
line3,=plt.plot(xc,tar,'-g')
line4,=plt.plot(xc,p1,'--r')
line5,=plt.plot(xc,p2,'-k')
line6, =plt.plot(xc,p3,'-c')
plt.legend((line1,line2,line3), ("Estimated Acceleration", "Predicted Acceleration",'Ground Truth'),loc=1)
plt.xlabel("Time(0.1s)")
plt.ylabel("Acceleration(feet/s^2)")
plt.show()


fig.savefig(fname1+fname2+'.eps',format='eps')
f = open(fname1+fname2+'.txt','w')
f.writelines([str(i)+'\n' for i in p])
f.close()





