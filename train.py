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
import copy

fname="C:\Users\Lenovo\Desktop\data\\2.8.xlsx"
bk=xlrd.open_workbook(fname)
sh = bk.sheet_by_name("Sheet1")
m=sh.nrows
n=sh.ncols
print m ,n
row_list = []

#获取各行数据
for i in range(0,m):
    row_data = sh.row_values(i)
    row=[]
    for j in range(0,n):
        row.append(float(row_data[j]))
    row_list.append(row)
row_list1=copy.deepcopy(row_list)
del row_list[0]
del row_list1[-1]
row_list2=np.column_stack((row_list,row_list1))

data=np.array(row_list2)
print data
da=data[:,[2,4,5,6,9,10,12,13,14]]
tar=data[:,[1]]
x=int(0.85*m)
da_train=da[0:x]
tar_train=tar[0:x]

da_test=da[x:]
tar_test=tar[x:]
est_gp = SymbolicRegressor(population_size=30000, init_depth=(2,10),
                           generations=30, stopping_criteria=0.01,
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

xc=np.arange(0.,float(len(tar))/10, 0.1)
fig = plt.figure()
fig.patch.set_facecolor('white')
line1, = plt.plot(xc,p,'-b')
line2,=plt.plot(xc,tar,'-g')
plt.legend((line1, line2), ("Our's Acceleration", 'Actual Acceleration'),loc=2)
plt.xlabel("Time(s)")
plt.ylabel("Acceleration(m/s^2)")
plt.show()