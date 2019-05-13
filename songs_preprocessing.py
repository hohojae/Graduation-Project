import utils
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
util = utils.Util()
songs = util.get_all_song()
songs_file_number = 3 # songs 파일 넘버 적음, 고칠 때 utils.py도 고쳐줘야함
file_path = "songs%d" % songs_file_number # songs파일 넘버로 설정
print("Load {} Songs...".format(len(songs)))
songs_name = ""
songs_len = []
songs_pitches = []
songs_durations = []
songs_list = []
songs_mean = []
songs_std = []
for song in songs:
    songs_name += "_" + song['name']
    songs_list.append(song['name'])
    songs_len.append(song['length'])
    songs_pitches.append(song['pitches'])
    songs_durations.append(song['durations'])
    print("name : ", song['name'])
    print("length : ", song['length'])
    print("pitches : ", song['pitches'])
    print("durations : ", song['durations'])
    print("")
# 곡 중간에 86이 들어간
for i in range(0, len(songs_pitches)):
    songs_pitches[i] = songs_pitches[i][1:]
    for k in range(0, len(songs_pitches[i])):
        if (songs_pitches[i][k] == "Rest"):
            os.remove(r"./midi/"+file_path+"/%s.mid" %songs_list[i])
            break
'''
# midi/songs/ 목록에서 곡의 중간에 'Rest'가 들어간 곡을 삭제함.
for i in range(0, len(songs_pitches)):
    songs_pitches[i] = songs_pitches[i][1:]
    for k in range(0, len(songs_pitches[i])):
        if (songs_pitches[i][k] == "Rest"):
            os.remove(r"./midi/"+file_path+"/%s.mid" %songs_list[i])
            break
            
# 곡들의 평균값과 표준편차값을 구함.
for i in range(0, len(songs_pitches)):
    songs_pitches[i] = songs_pitches[i][1:]
    songs_mean.append(np.mean(songs_pitches[i]))
    songs_std.append(np.std(songs_pitches[i]))
with open("./mean_std/mean_std_1.csv", "w", newline='') as f:
    wr = csv.writer(f)
    wr.writerow(songs_mean)
    wr.writerow(songs_std)
'''
'''
# 평균값과 표준편차값을 산포도로 저장함.
plt.scatter(songs_mean, songs_std, c="black")
plt.grid()
plt.title("%dSongs Mean and Standard Deviation Distribution" % len(songs_pitches))
plt.xlabel("Mean")
plt.ylabel("Std")
plt.xticks([40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) # x축의 범위 지정 40~90
plt.savefig("songs_mean_std_scatter%d.png" % songs_file_number)
plt.show()
'''