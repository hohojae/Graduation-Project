import matplotlib.pyplot as plt
import os
import csv
import numpy as np
x_list = []
y_list = []
x, y = [], []
with open('./mean_std/mean_std.csv', 'r') as f:
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
    print(x_list, y_list)


plt.scatter(x_list, y_list, c="black", label='data1') # 산포도 그려서 저장
plt.title("RNN model Mean and Standard Deviation Distribution")
plt.grid()
plt.xlabel("Mean")
plt.ylabel("Std")
plt.legend(loc='upper left')
plt.xticks([40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) # x축의 범위 지정 40~90
plt.savefig("./mean_std/mean_std_test.eps")
plt.show()