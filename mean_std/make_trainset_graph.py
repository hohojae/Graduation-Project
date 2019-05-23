import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm


with open("mean_std_1.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x1 = [float(i) for i in mylist[0]]
    y1 = [float(i) for i in mylist[1]]
with open("mean_std_2.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x2 = [float(i) for i in mylist[0]]
    y2 = [float(i) for i in mylist[1]]
with open("mean_std_3.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x3 = [float(i) for i in mylist[0]]
    y3 = [float(i) for i in mylist[1]]
with open("mean_std_4.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x4 = [float(i) for i in mylist[0]]
    y4 = [float(i) for i in mylist[1]]
with open("mean_std_5.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x5 = [float(i) for i in mylist[0]]
    y5 = [float(i) for i in mylist[1]]
with open("mean_std_6.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x6 = [float(i) for i in mylist[0]]
    y6 = [float(i) for i in mylist[1]]
with open("mean_std_7.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x7 = [float(i) for i in mylist[0]]
    y7 = [float(i) for i in mylist[1]]
with open("mean_std_8.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x8 = [float(i) for i in mylist[0]]
    y8 = [float(i) for i in mylist[1]]
with open("mean_std_9.csv", "r") as f:
    rd = csv.reader(f)
    mylist = list(rd)
    x9 = [float(i) for i in mylist[0]]
    y9 = [float(i) for i in mylist[1]]

q = np.arange(9)
ys = [i+q+(i*q)**2 for i in range(9)]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

x_mean = np.mean(x1,x2,x3,x4,x5)
x_mean2 = np.mean(x6,x7,x8,x9)
x_mean3 = np.mean(x_mean, x_mean2)

y_mean = np.mean(y1,y2,y3,y4,y5)
y_mean2 = np.mean(y6,y7,y8,y9)
y_mean3 = np.mean(y_mean, y_mean2)
print(x_mean3, y_mean3)
'''
plt.figure(1)
plt.title("9Class Dataset Mean and Standard Deviation Distribution")
plt.grid()
plt.xlabel("Mean")
plt.ylabel("Std")
plt.scatter(x1, y1, c=colors[0], label="class1") # 1
plt.scatter(x2, y2, c=colors[1], label="class2") # 2
plt.scatter(x3, y3, c=colors[2], label="class3") # 3
plt.scatter(x4, y4, c=colors[3], label="class4") # 4
plt.scatter(x5, y5, c=colors[4], label="class5") # 5
plt.scatter(x6, y6, c=colors[5], label="class6") # 6
plt.scatter(x7, y7, c=colors[6], label="class7") # 7
plt.scatter(x8, y8, c=colors[7], label="class8") # 8
plt.scatter(x9, y9, c=colors[8], label="class9") # 9

plt.legend(loc='upper left')
plt.xticks([40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) # x축의 범위 지정 40~90
plt.yticks([2, 3, 4, 5, 6, 7]) # y축
plt.savefig("./mean_std_dataset.png")
'''