#!/usr/bin/env python
# coding: utf-8

# In[70]:


import csv
import numpy as np
import matplotlib.pyplot as plt



# In[71]:


with open("./songs/songs.csv", "r") as f:
    rd = csv.reader(f)
    class1 = list(rd)
    x1 = []
    x_1= []
    for i in range(0, len(class1)):
        if (i % 2) == 0:
            x_1.append(class1[i])
    for t in x_1:
        t = float(t[0])
        x1.append(t)
    x1 = np.round(x1, 6)
'''
with open("./cos_sim/songs2/songs2.csv", "r") as f:
    rd = csv.reader(f)
    class2 = list(rd)
    x2 = []
    x_2= []
    for i in range(0, len(class2)):
        if (i % 2) == 0:
            x_2.append(class2[i])
    for t in x_2:
        t = float(t[0])
        x2.append(t)
    x2 = np.round(x2, 6)
with open("./cos_sim/songs3/songs3.csv", "r") as f:
    rd = csv.reader(f)
    class3 = list(rd)
    x3 = []
    x_3= []
    for i in range(0, len(class3)):
        if (i % 2) == 0:
            x_3.append(class3[i])
    for t in x_3:
        t = float(t[0])
        x3.append(t)
    x3 = np.round(x3, 6)
with open("./cos_sim/songs4/songs4.csv", "r") as f:
    rd = csv.reader(f)
    class4 = list(rd)
    x4 = []
    x_4= []
    for i in range(0, len(class4)):
        if (i % 2) == 0:
            x_4.append(class4[i])
    for t in x_4:
        t = float(t[0])
        x4.append(t)
    x4 = np.round(x4, 6)
with open("./cos_sim/songs5/songs5.csv", "r") as f:
    rd = csv.reader(f)
    class5 = list(rd)
    x5 = []
    x_5= []
    for i in range(0, len(class5)):
        if (i % 2) == 0:
            x_5.append(class5[i])
    for t in x_5:
        t = float(t[0])
        x5.append(t)
    x5 = np.round(x5, 6)
with open("./cos_sim/songs6/songs6.csv", "r") as f:
    rd = csv.reader(f)
    class6 = list(rd)
    x6 = []
    x_6= []
    for i in range(0, len(class6)):
        if (i % 2) == 0:
            x_6.append(class6[i])
    for t in x_6:
        t = float(t[0])
        x6.append(t)
    x6 = np.round(x6, 6)
with open("./cos_sim/songs7/songs7.csv", "r") as f:
    rd = csv.reader(f)
    class7 = list(rd)
    x7 = []
    x_7= []
    for i in range(0, len(class7)):
        if (i % 2) == 0:
            x_7.append(class7[i])
    for t in x_7:
        t = float(t[0])
        x7.append(t)
    x7 = np.round(x7, 6)
with open("./cos_sim/songs8/songs8.csv", "r") as f:
    rd = csv.reader(f)
    class8 = list(rd)
    x8 = []
    x_8= []
    for i in range(0, len(class8)):
        if (i % 2) == 0:
            x_8.append(class8[i])
    for t in x_8:
        t = float(t[0])
        x8.append(t)
    x8 = np.round(x8, 6)
with open("./cos_sim/songs9/songs9.csv", "r") as f:
    rd = csv.reader(f)
    class9 = list(rd)
    x9 = []
    x_9= []
    for i in range(0, len(class9)):
        if (i % 2) == 0:
            x_9.append(class9[i])
    for t in x_9:
        t = float(t[0])
        x9.append(t)
    x9 = np.round(x9, 6)
len(x1)
'''

# In[108]:


#plt.figure(figsize=(30,7))
#1
#plt.subplot(1,3,1)
ys, xs, patches = plt.hist(x1, bins=5, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Whole Data Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
plt.savefig("cos_sim_pearson_RNN_whole_data.png")
#2
'''
plt.subplot(1,3,2)
ys, xs, patches = plt.hist(x2, bins=5, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 2 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
#3
plt.subplot(1,3,3)
ys, xs, patches = plt.hist(x3, bins=5, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 3 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
plt.savefig("cos_sim_RNN1.png")

#plt.subplots_adjust(hspace = 15, wspace = 0)


# In[109]:


plt.figure(figsize=(30,7))
#4
plt.subplot(1,3,1)
ys, xs, patches = plt.hist(x4, bins=5, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 4 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
#5
plt.subplot(1,3,2)
ys, xs, patches = plt.hist(x5, bins=5, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 5 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
#6
plt.subplot(1,3,3)
ys, xs, patches = plt.hist(x6, bins=3, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 6 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
plt.savefig("cos_sim_RNN2.png")


# In[110]:


plt.figure(figsize=(30,7))
#7
plt.subplot(1,3,1)
ys, xs, patches = plt.hist(x7, bins=3, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 7 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
#8
plt.subplot(1,3,2)
ys, xs, patches = plt.hist(x8, bins=2, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 8 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
#9
plt.subplot(1,3,3)
ys, xs, patches = plt.hist(x9, bins=2, density=True, cumulative=True, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='b', alpha=0.8)
plt.title("Class 9 Cosine Similarity")
plt.grid()
plt.xlabel("Cosine Similarity")
plt.ylabel("Count")
plt.xlim(0.9955, 1.0)
plt.xticks([0.9955, 0.9960, 0.9965, 0.9970, 0.9975, 0.9980, 0.9985, 0.9990, 0.9995, 1.0000]) # x축의 범위 지정 40~90
plt.savefig("cos_sim_RNN3.png")


# In[78]:





# In[ ]:



'''
