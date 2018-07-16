#coding=utf-8
import os
import matplotlib.pyplot as plt
import xlrd
import numpy as np
import math
index = '1247.1269'
file1 = 'C:\Users\Lenovo\Desktop\data\carcarcar\\result_change\\lane_changing0750.'+index
simulated_data=[]
for line in open(file1+'.txt'):
	simulated_data.append(float(line),)
simulated_data = np.array(simulated_data)
t = len(simulated_data)
print t
fname1='C:\\Users\\Lenovo\\Desktop\\data\\lane_changing0750'
real_data = np.loadtxt(fname1+'.txt')
real_data = np.array(real_data)
m=1247
n=1269
real_data = real_data[m:n,:]
a_real = real_data[:,3]
a_real = np.array(a_real)
v_real = real_data[:,4]
for i in range(len(a_real)-1):
	v_real[i+1] = v_real[i] + 0.1*a_real[i]
v_real = np.array(v_real)
v_follower = real_data[:,13]
v_follower = np.array(v_follower)
s_followreal = real_data[:,15]
for i in range(len(v_real)-1):
	s_followreal[i+1] = s_followreal[i] + (v_real[i]-v_follower[i])*0.1
v_leader = real_data[:,12]
s_leadreal = real_data[:,14]
for i in range(len(v_real)-1):
	s_leadreal[i+1] = s_leadreal[i] + (-v_real[i]+v_leader[i])*0.1

v_simulated = [None]*t
v_simulated[0] = v_real[0]
for j in range(1,t):
	v_simulated[j] = simulated_data[j-1]*0.1+v_simulated[j-1]
	#v_simulated[j] = simulated_data[j-1]*0.1+ v_real[j-1]
s_followsimulated = [None]*t
s_followsimulated[0] = s_followreal[0]
for i in range(1,t):
	s_followsimulated[i] = s_followsimulated[i-1] + (-v_follower[i-1]*0.1 + v_simulated[i-1] *0.1)
	#s_simulated[i] = s_real[i-1] - (v_follower[i-1]*0.1 - v_simulated[i-1] *0.1)
s_leadsimulated = [None]*t
s_leadsimulated[0] = s_leadreal[0]
for i in range(1,t):
	s_leadsimulated[i] = s_leadsimulated[i-1] + (v_leader[i-1]*0.1 - v_simulated[i-1] *0.1)
	#s_simulated[i] = s_real[i-1] - (v_follower[i-1]*0.1 - v_simulated[i-1] *0.1)

Smin=2.5
amax=4.0 
Vd=20
T = 0.1
bcom = 3.0 
def IDM(vl,al,S,speed):
    vl = vl / 3.28
    al = al / 3.28
    S = S / 3.28
    speed = speed / 3.28
    vq = speed - vl
    s1 = Smin + speed*T + speed*vq / (2 * ((amax*bcom)**0.5))
    Aidm = amax*(1 - (speed / Vd)*(speed / Vd)*(speed / Vd)*(speed / Vd)) - bcom*(s1 / S)*(s1 / S)
    
    if al>amax :
        al1 =amax
    else:
        al1 =al
    
    if (vl*vq) <= (-2 * S*al1):
        Acah = speed*speed*al1 / (vl*vl - 2 * S*al1)
    else:
        Acah = al1*al1 - vq*vq / (2 * S)

    Aadjust = Acah + bcom*math.tanh((Aidm - Acah) / bcom)

    if Aidm >= Acah :
        Afinal = Aidm
    else:
        Afinal = 0.01*Aidm + 0.99*Aadjust
    
    if Afinal > 20:
    	Afinal = 20
    print speed
    print s1,S
    return Aidm * 3.28

a_idm_simulated = [None]*t
v_idm_simulated = [None]*t
s_idm_simulated = [None]*t
a_idm_simulated[0] = real_data[0][3]
v_idm_simulated[0] = real_data[0][4]
s_idm_simulated[0] = real_data[0][14]
for i in range(1,t):
	if i!=t-1:
		a_idm_simulated[i] = IDM(real_data[i][12],(real_data[i+1][12]-real_data[i][12])*10,real_data[i][14],real_data[i][4])
		v_idm_simulated[i] = v_idm_simulated[i-1] + 0.1*a_idm_simulated[i-1]
		s_idm_simulated[i] = s_idm_simulated[i-1] + (real_data[i-1][12]-v_idm_simulated[i-1])*0.1
	else:
		a_idm_simulated[i] = IDM(real_data[i][12],(real_data[i][12]-real_data[i-1][12])*10,real_data[i][14],real_data[i][4])
		v_idm_simulated[i] = v_idm_simulated[i-1] + 0.1*a_idm_simulated[i-1]
		s_idm_simulated[i] = s_idm_simulated[i-1] + (real_data[i-1][12]-v_idm_simulated[i-1])*0.1
#print a_idm_simulated

Sde = 53.3
bcom = 4 
def cal(s, car1, car2):
#houche qianche
    s = s / 3.28
    car1 = car1 / 3.28
    car2 = car2 / 3.28
    x = 0.3
    y = 0.8
    z = 1.5
    t1 = 0.1
    t2 = 0.5
    t3 = 1
    t4 = 1.5
    if s < Sde - z:
        t = t1
    elif s >Sde - z and s <= Sde - y:
        t = t2
    elif s>Sde - y and s <= Sde - x:
        t = t3
    elif s > Sde - x and s <= Sde + x:
        t = t4
    elif s > Sde + x and s < Sde + y:
        t = t3
    elif s > Sde + y and s < Sde + z:
        t = t2
    else:
        t = t1
    #t =  1
    vq = car1 - car2
    #vq = abs(vq)
    s1 = Sde +vq*car2 / (2 * math.sqrt(amax*bcom))
    print s, s1,car1, car2,t
    print 1-(car2 / (Sde / t))**4,(s / s1)**2-1
    a = amax*(1 - (car2 / (Sde / t))**4) - bcom*((s / s1)**2)
    if a > 15:
    	a = 15
    return a*3.28
a_yang_simulated = [None]*t
v_yang_simulated = [None]*t
s_yang_simulated = [None]*t
a_yang_simulated[0] = real_data[0][3]
v_yang_simulated[0] = real_data[0][4]
s_yang_simulated[0] = real_data[0][15]
for i in range(1,t):
	a_yang_simulated[i] = cal(real_data[i][15],real_data[i][13],real_data[i][4])
	v_yang_simulated[i] = v_yang_simulated[i-1] + 0.1*a_yang_simulated[i-1]
	# if v_yang_simulated[i] > 50:
	# 	v_yang_simulated[i] = 50
	s_yang_simulated[i] = s_yang_simulated[i-1] + (v_yang_simulated[i-1]-real_data[i-1][13])*0.1

xc = range(0,t)
fig4 = plt.figure()
fig4.patch.set_facecolor('white')
x = int(0.66*t)
line14, = plt.plot(xc[0:x],simulated_data[0:x],'-r')
line15, = plt.plot(xc[x-1:],simulated_data[x-1:],'--r')
line16, = plt.plot(xc,a_real,'-g')
line18, = plt.plot(xc,a_yang_simulated,'-c')
line19, = plt.plot(xc,a_idm_simulated,'-k')
plt.legend((line14,line15,line16,line18,line19),('Estimated Acceleration','Predicted Acceleration','Ground Truth','Yang\'s method','Jin\'s method'),loc=4)
plt.xlabel('Time(0.1s)')
plt.ylabel('Acceleration(feet/s^2)')
plt.show()