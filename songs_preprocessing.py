import utils
import numpy as np
import os
import matplotlib.pyplot as plt
util = utils.Util()
songs = util.get_all_song()

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
# midi/songs/ 목록에서 곡의 중간에 'Rest'가 들어간 곡을 삭제함.
for i in range(0, len(songs_pitches)):
    songs_pitches[i] = songs_pitches[i][1:]
    for k in range(0, len(songs_pitches[i])):
        if (songs_pitches[i][k] == "Rest"):
            os.remove(r"./midi/songs/%s.mid" %songs_list[i])
            break
# 곡들의 평균값과 표준편차값을 구함.
for i in range(0, len(songs_pitches)):
    songs_pitches[i] = songs_pitches[i][1:]
    songs_mean.append(np.mean(songs_pitches[i]))
    songs_std.append(np.std(songs_pitches[i]))
# 평균값과 표준편차값을 산포도로 저장함.
plt.scatter(songs_mean, songs_std, c="black")
plt.grid()
plt.title("%dSongs Mean and Standard Deviation Distribution" % len(songs_pitches))
plt.xlabel("Mean")
plt.ylabel("Std")
plt.xticks([40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]) # x축의 범위 지정 40~90
plt.savefig("songs_mean_std_scatter.png")
plt.show()