import matplotlib.pyplot as plt
import os
import csv
import numpy as np
x_list = [] # 코사인 유사도
y_list = [] # 피어슨 상관계수
x, y = [], []
with open('./songs/cos_sim_pearson.csv', 'r') as f:
    rdr = csv.reader(f)
    mylist = list(rdr)

    for i in range(0, len(mylist)):
        if (i % 2) == 0:
            x.append(mylist[i])
        else:
            y.append(mylist[i])
    for t in x:
        t = float(t[0])
        x_list.append(t)
    for i in y:
        i = float(i[0])
        y_list.append(i)

x = np.round(x_list, 3)
y = np.round(y_list, 3)
print(x, y)

plt.figure(figsize=(10, 6))
ys, xs, patches = plt.hist(x, bins=5, density=True, cumulative=False, histtype='bar',
                          orientation='vertical', rwidth=0.8,
                          color='black')
plt.title("RNN model Cosine Similarity and Pearson Correlation Histogram")
plt.grid()
plt.xlabel("Mean")
plt.ylabel("Std")
plt.xticks([40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) # x축의 범위 지정 40~90
plt.savefig("cos_sim_pearson_RNN_whole_data.png")
plt.show()
