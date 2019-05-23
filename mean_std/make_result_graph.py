import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
'''
with open("./songs1/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class1 = list(rd)
    x1, y1 = [], []
    x_1, y_1 = [], []
    for i in range(0, len(class1)):
        if (i % 2) == 0:
            x_1.append(class1[i])
        else:
            y_1.append(class1[i])
    for t in x_1:
        t = float(t[0])
        x1.append(t)
    for i in y_1:
        i = float(i[0])
        y1.append(i)
with open("./songs2/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class2 = list(rd)
    x2, y2 = [], []
    x_2, y_2 = [], []
    for i in range(0, len(class2)):
        if (i % 2) == 0:
            x_2.append(class2[i])
        else:
            y_2.append(class2[i])
    for t in x_2:
        t = float(t[0])
        x2.append(t)
    for i in y_2:
        i = float(i[0])
        y2.append(i)
with open("./songs3/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class3 = list(rd)
    x3, y3 = [], []
    x_3, y_3 = [], []
    for i in range(0, len(class3)):
        if (i % 2) == 0:
            x_3.append(class3[i])
        else:
            y_3.append(class3[i])
    for t in x_3:
        t = float(t[0])
        x3.append(t)
    for i in y_3:
        i = float(i[0])
        y3.append(i)
with open("./songs4/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class4 = list(rd)
    x4, y4 = [], []
    x_4, y_4 = [], []
    for i in range(0, len(class4)):
        if (i % 2) == 0:
            x_4.append(class4[i])
        else:
            y_4.append(class4[i])
    for t in x_4:
        t = float(t[0])
        x4.append(t)
    for i in y_4:
        i = float(i[0])
        y4.append(i)
with open("./songs5/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class5 = list(rd)
    x5, y5 = [], []
    x_5, y_5 = [], []
    for i in range(0, len(class5)):
        if (i % 2) == 0:
            x_5.append(class5[i])
        else:
            y_5.append(class5[i])
    for t in x_5:
        t = float(t[0])
        x5.append(t)
    for i in y_5:
        i = float(i[0])
        y5.append(i)
with open("./songs6/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class6 = list(rd)
    x6, y6 = [], []
    x_6, y_6 = [], []
    for i in range(0, len(class6)):
        if (i % 2) == 0:
            x_6.append(class6[i])
        else:
            y_6.append(class6[i])
    for t in x_6:
        t = float(t[0])
        x6.append(t)
    for i in y_6:
        i = float(i[0])
        y6.append(i)
with open("./songs7/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class7 = list(rd)
    x7, y7 = [], []
    x_7, y_7 = [], []
    for i in range(0, len(class7)):
        if (i % 2) == 0:
            x_7.append(class7[i])
        else:
            y_7.append(class7[i])
    for t in x_7:
        t = float(t[0])
        x7.append(t)
    for i in y_7:
        i = float(i[0])
        y7.append(i)
with open("./songs8/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class8 = list(rd)
    x8, y8 = [], []
    x_8, y_8 = [], []
    for i in range(0, len(class8)):
        if (i % 2) == 0:
            x_8.append(class8[i])
        else:
            y_8.append(class8[i])
    for t in x_8:
        t = float(t[0])
        x8.append(t)
    for i in y_8:
        i = float(i[0])
        y8.append(i)
with open("./songs9/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class9 = list(rd)
    x9, y9 = [], []
    x_9, y_9 = [], []
    for i in range(0, len(class9)):
        if (i % 2) == 0:
            x_9.append(class9[i])
        else:
            y_9.append(class9[i])
    for t in x_9:
        t = float(t[0])
        x9.append(t)
    for i in y_9:
        i = float(i[0])
        y9.append(i)
'''
with open("./songs/mean_std.csv", "r") as f:
    rd = csv.reader(f)
    class_ = list(rd)
    x, y = [], []
    x_, y_ = [], []
    for i in range(0, len(class_)):
        if (i % 2) == 0:
            x_.append(class_[i])
        else:
            y_.append(class_[i])
    for t in x_:
        t = float(t[0])
        x.append(t)
    for i in y_:
        i = float(i[0])
        y.append(i)

#q = np.arange(9)
#ys = [i+q+(i*q)**2 for i in range(9)]
#colors = cm.rainbow(np.linspace(0, 1, len(ys)))

plt.figure(1)
plt.title("RNN model Mean and Standard Deviation Distribution")
plt.grid()
plt.xlabel("Mean") # legend 추가하기
plt.ylabel("Std")
plt.scatter(x, y, c='b', label="Whole Data") # 1
'''
plt.scatter(x2, y2, c=colors[1], label="class2") # 2
plt.scatter(x3, y3, c=colors[2], label="class3") # 3
plt.scatter(x4, y4, c=colors[3], label="class4") # 4
plt.scatter(x5, y5, c=colors[4], label="class5") # 5
plt.scatter(x6, y6, c=colors[5], label="class6") # 6
plt.scatter(x7, y7, c=colors[6], label="class7") # 7
plt.scatter(x8, y8, c=colors[7], label="class8") # 8
plt.scatter(x9, y9, c=colors[8], label="class9") # 9
'''
plt.legend(loc='upper left')
plt.xticks([40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) # x축의 범위 지정 40~90
plt.yticks([2, 3, 4, 5, 6, 7]) # y축
plt.savefig("mean_std_result_whole_data.png")
plt.show()