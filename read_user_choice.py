import os

def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print (full_filename)

with open('./select_music.txt', 'r') as f:
    rd = f.read()
    songs_list = rd.split()

user_choice = []

for i in range(0, len(songs_list)):
    user_choice.append(int(songs_list[i][:-1])) # [1, 3, 5, 10 ,11]의 형식



print(user_choice)
#for i in range(0, len(user_choice)):
#    search("./midi/test/%d.mid"%user_choice[i])
