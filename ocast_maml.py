import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random

batchsz = 10000  # batch of set, not batch of imgs
n_way = 5  # n-way
k_shot = 5  # k-shot
k_query = 15  # for evaluation
setsz = n_way * k_shot  # num of samples per set
querysz = n_way * k_query  # number of samples per set for evaluation
resize = 84  # resize to
startidx = 0  # index label not from 0, but from startidx

transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                     transforms.Resize((resize, resize)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])

root = '/home/atik/Documents/Ocast/borescope-adr-lm2500-data-develop/Processed/wo_Dup/'
path = os.path.join(root, 'test/')  # image path
 
def read_csv(csvf):
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            print(i)
            filename = row[0]
            label = row[1]
            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels
    
dictLabels = read_csv(os.path.join(file))  
    
data = []
img2label = {}
for i, (label, imgs) in enumerate(dictLabels.items()):
    data.append(imgs)  # [[img1, img2, ...], [img111, ...]]
    img2label[label] = i + startidx  # {"img_name[:9]":label}
cls_num = len(data)


from os import walk

filenames = next(walk(path))[1]

import glob

img = []
for i in range(len(filenames)):
    for images in glob.iglob(f'{path+filenames[0]}/*'):
        # check if the image ends with png
        if (images.endswith(".jpeg")):
            img_temp = images[len(path+filenames[0]+'/'):]
            img.append(img_temp)

 









