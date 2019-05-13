import utils
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
util = utils.Util()
songs = util.get_all_song()

with open("mean_std_9.csv", "r") as f:
    rd = csv.reader(f)



x_list = []
y_list = []
x, y = [], []
with open('mean_std.csv', 'r') as f:
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