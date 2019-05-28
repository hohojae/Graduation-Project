from glob import glob
import subprocess
# ver1        *   *    *   *   *    *    *  *   *
# ver2        *     *    *   *  *   *    *  *   *
data_class = [420, 309, 65, 52, 37, 26, 15, 11, 10] # songs1_num, songs2_num, ... , songs9_num
'''
f_list = glob('train.py')
f_list.append('test.py')
for i in range(0, 3):
    for f in f_list:
        subprocess.call(['python', f])
'''

testfile = glob('test.py')
for i in range(0, data_class[6]):
    subprocess.call(['python', testfile[0]])
    print("%d is tested!" % i)
