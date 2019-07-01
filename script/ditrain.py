import os
import random

trainval_percent = 0.9
train_percent = 0.6
xmlfilepath = 'annotation'
txtsavepath = 'imagesets\main'
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open('imagesets/main/trainval.txt', 'w')
ftest = open('imagesets/main/test.txt', 'w')
ftrain = open('imagesets/main/train.txt', 'w')
fval = open('imagesets/main/val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
