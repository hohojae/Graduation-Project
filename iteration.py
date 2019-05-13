from glob import glob
import subprocess
#             *   *    *   *   *    *    *  *   *
data_class = [37, 65, 309, 26, 15, 420, 10, 52, 11] # songs1_num, songs2_num, ... , songs9_num
'''
f_list = glob('train.py')
f_list.append('test.py')
for i in range(0, 3):
    for f in f_list:
        subprocess.call(['python', f])
'''

testfile = glob('test.py')
for i in range(0, data_class[2]):
    subprocess.call(['python', testfile[0]])
    print("%d is tested!" % i)
