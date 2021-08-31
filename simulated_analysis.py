import pandas as pd
import numpy as np

import helpers as H
import dit
import matplotlib.pyplot as plt
import time
import numpy as np
import random


# Generating Simulated Data
def sample_data(n=10000):
    data = []
    
    zb = np.random.binomial(1,0.5,(n,))  #Sensitive Attribute
    
    
    ug1 =  np.random.binomial(1,0.9,(n,)) #Latent
    ug2 =  np.random.binomial(1,0.1,(n,)) #Latent
    ug3 =  np.random.binomial(1,0.5,(n,)) #Latent
    
    #ug1=  np.random.normal(0,0.5,(n,))
    #ug1=  np.random.normal(0,0.5,(n,))
    #ug2 = np.random.normal(0,0.5,(n,))
    
    for i in range(n):
        xb=ug1[i]
        xg1=zb[i]+ug3[i]
        xg2=ug3[i]
        y=xb+xg1+xg2+ug2[i] #ug2[i] is additional noise
    
        data.append([xb,xg1,xg2,y,zb[i]])


    return np.array(data)

# Data Analysis
D = sample_data(10000)
X = D[:,0:3]
Y = D[:,3:4]
Z = D[:,4:5]


print("Shape of Feature matrix X is:", np.shape(X))
print("Features are: GPA, GRE, Reco")
features=['GPA','GRE','Reco','Total']
#print(X[0:5,:])
#print(X[0:5,:])
#print(Y[0:5])

print("Sample Data Visualization")
print(D[0:5,:])


#Contribution of features: 1D
n_features=3
contri={}

for i in range(n_features):
    Uni, Red, MI = H.PID1(Z,Y,X[:,i])
    contri["{}".format(i)]=np.around(Red,3)


#Contribution of features: 2D
for i in range(n_features):
    for j in range(i):
        if i!=j:
            Uni, Red, MI = H.PID2(Z,Y,X[:,[i,j]])
            contri["{}{}".format(i,j)]=np.around(Red,3)


Uni, Red, MI = H.PID3(Z,Y,X[:,:])
contri["210"]=np.around(Red,3)

print("Contribution Calculated:",contri)

print("SANITY CHECK")
print("Actual MI (should match contri['210']):",MI)

contri_adm=[]
c0=(1/3)*(contri['0']+contri['210']-contri['21'])+(1/6)*(contri['10']+contri['20']-contri['1']-contri['2'])

c1=(1/3)*(contri['1']+contri['210']-contri['20'])+(1/6)*(contri['10']+contri['21']-contri['0']-contri['2'])

c2=(1/3)*(contri['2']+contri['210']-contri['10'])+(1/6)*(contri['20']+contri['21']-contri['0']-contri['1'])

contri_adm.append(c0)
contri_adm.append(c1)
contri_adm.append(c2)
contri_adm.append(MI)

fig1 = plt.figure()

# creating the bar plot
plt.bar(features, contri_adm, color ='maroon',
width = 0.4)
plt.ylim(0,0.5)
plt.xlabel("Features")
plt.ylabel("Contributions")
plt.title("Contributions of Individual Features to Overall Disparity")
plt.show()


fig2 = plt.figure()

# creating the bar plot
F=['X1','X2','X3','X1,X2','X1,X3','X2,X3','X1,X2,X3']

plt.bar(F, contri.values(), color ='maroon',
        width = 0.3)
plt.ylim(0,0.5)
plt.xlabel("Features")
plt.ylabel("Redundant Information")
plt.title("Redundant Information Across Different Sets of Features")
plt.show()




