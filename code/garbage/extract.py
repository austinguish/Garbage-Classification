import os
import numpy as np
origin_dir = "./Garbageclassification/"
train_dir = "./train_set/"
valid_dir = "./valid_set/"
test_dir = "./test_set/"
num_class = {1:"glass",2:"paper",3:"cardboard",4:"plastic",5:"metal",6:"trash"}  
train_file_list = "one-indexed-files-notrash_train.txt"
valid_file_list = "one-indexed-files-notrash_val.txt"
test_file_list = "one-indexed-files-notrash_test.txt"
cursor1 = np.loadtxt(train_file_list,dtype="str")
cursor2 = np.loadtxt(valid_file_list,dtype="str")
cursor3 = np.loadtxt(test_file_list,dtype="str")
for i in range(len(cursor2)):
    file_dir = origin_dir+num_class[int(cursor2[i][1])]+'/'+cursor2[i][0]
    print(file_dir)
    target_dir =  valid_dir+num_class[int(cursor2[i][1])]+'/'
    os.system("cp %s %s" % (file_dir, target_dir))

for i in range(len(cursor3)):
    file_dir = origin_dir+num_class[int(cursor3[i][1])]+'/'+cursor3[i][0]
    print(file_dir)
    target_dir =  test_dir+num_class[int(cursor3[i][1])]+'/'
    os.system("cp %s %s" % (file_dir, target_dir))
    