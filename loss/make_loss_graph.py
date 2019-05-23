#!/usr/bin/env python
# coding: utf-8

# In[12]:


import csv
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


with open("./songs1/songs1.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y1 = [float(i) for i in mylist[0]]
    x1 = [float(i) for i in mylist[1]]
    y1 = np.array(y1)
with open("./songs2/songs2.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y2 = [float(i) for i in mylist[0]]
    x2 = [float(i) for i in mylist[1]]
    y2 = np.array(y2)
with open("./songs3/songs3.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y3 = [float(i) for i in mylist[0]]
    x3 = [float(i) for i in mylist[1]]
    y3 = np.array(y3)
with open("./songs4/songs4.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y4 = [float(i) for i in mylist[0]]
    x4 = [float(i) for i in mylist[1]]
    y4 = np.array(y4)
with open("./songs5/songs5.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y5 = [float(i) for i in mylist[0]]
    x5 = [float(i) for i in mylist[1]]
    y5 = np.array(y5)
with open("./songs6/songs6.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y6 = [float(i) for i in mylist[0]]
    x6 = [float(i) for i in mylist[1]]
    y6 = np.array(y6)
with open("./songs7/songs7.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y7 = [float(i) for i in mylist[0]]
    x7 = [float(i) for i in mylist[1]]
    y7 = np.array(y7)
with open("./songs8/songs8.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y8 = [float(i) for i in mylist[0]]
    x8 = [float(i) for i in mylist[1]]
    y8 = np.array(y8)
with open("./songs9/songs9.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y9 = [float(i) for i in mylist[0]]
    x9 = [float(i) for i in mylist[1]]
    y9 = np.array(y9)



# In[46]:


a = np.vstack([y1,y2,y3,y4,y5,y6,y7,y8,y9])

avgloss = []
stdloss = []
loss_max = []
loss_min = []
for i in range(0, len(y1)):
    b=np.mean([a[0][i],a[1][i],a[2][i],a[3][i],a[4][i]])
    c = np.mean([a[5][i],a[6][i],a[7][i],a[8][i]])
    avgloss.append(np.mean([b,c]))
for i in range(0, len(y1)):
    b=np.std([a[0][i],a[1][i],a[2][i],a[3][i],a[4][i]])
    c = np.std([a[5][i],a[6][i],a[7][i],a[8][i]])
    stdloss.append(np.std([b,c]))
for i in range(0, len(y1)):
    loss_max.append(avgloss[i]+stdloss[i])
    loss_min.append(avgloss[i]-stdloss[i])

'''
# In[52]:
with open("./songs/loss_17000705.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    y1 = [float(i) for i in mylist[0]]
    x1 = [float(i) for i in mylist[1]]
    y1 = np.array(y1)
'''
step = x1

plt.plot(step, avgloss)
plt.title("RNN Model Loss")
plt.grid()
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
plt.fill_between(step, loss_min, loss_max, color = 'b', alpha = 0.3)
plt.savefig("loss_RNN_9class_mean_std.png")


# In[ ]:




