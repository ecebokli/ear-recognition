import numpy as np
import os
import cv2
import time
import csv
import shutil

f_train = open("awecrop\\train.txt", "r")
f_test = open("awecrop\\test.txt", "r")

lines = f_train.read().splitlines() 
train = []
for line in lines:
    splits = line.split(" ")
    for split in splits:
        train.append(split)
#print(train)
class_names = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100']

for i in range(0,100):
    for j in range(1,11):
        number = i*10+j
        j = "{0:0=2d}".format(j)
        
        if str(number) in train:
            os.makedirs(os.path.dirname('custom/AWE/train/'+class_names[i]+'/'+j+'.png'), exist_ok=True)
            shutil.copy2('awecrop/'+class_names[i]+'/'+j+'.png', 'custom/AWE/train/'+class_names[i]+'/'+j+'.png')
        else:
            os.makedirs(os.path.dirname('custom/AWE/test/'+class_names[i]+'/'+j+'.png'), exist_ok=True)
            shutil.copy2('awecrop/'+class_names[i]+'/'+j+'.png', 'custom/AWE/test/'+class_names[i]+'/'+j+'.png')