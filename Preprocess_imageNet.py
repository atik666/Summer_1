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

root = '/home/atik/Documents/MAML/Summer_1/datasets/miniImageNet/mini-imagenet'
path = os.path.join(root, 'images')  # image path
file = os.path.join(root, 'train' + '.csv')
 
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

support_x_batch = []  # support set batch
query_x_batch = []  # query set batch
for b in range(batchsz):  # for each batch
    # 1.select n_way classes randomly
    selected_cls = np.random.choice(cls_num, n_way, False)  # no duplicate
    np.random.shuffle(selected_cls)
    support_x = []
    query_x = []
    for cls in selected_cls:
        # 2. select k_shot + k_query for each class
        selected_imgs_idx = np.random.choice(len(data[cls]), k_shot + k_query, False)
        np.random.shuffle(selected_imgs_idx)
        indexDtrain = np.array(selected_imgs_idx[:k_shot])  # idx for Dtrain
        indexDtest = np.array(selected_imgs_idx[k_shot:])  # idx for Dtest
        support_x.append(
            np.array(data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
        query_x.append(np.array(data[cls])[indexDtest].tolist())

    # shuffle the correponding relation between support set and query set
    random.shuffle(support_x)
    random.shuffle(query_x)

    support_x_batch.append(support_x)  # append set to current sets
    query_x_batch.append(query_x)  # append sets to current sets
    
    
support_x = torch.FloatTensor(setsz, 3, resize, resize)

support_y = np.zeros((setsz), dtype=np.int32)    

query_x = torch.FloatTensor(querysz, 3, resize, resize)

query_y = np.zeros((querysz), dtype=np.int32)
    
for index in range(batchsz):
    
    flatten_support_x = [os.path.join(path, item)
                     for sublist in support_x_batch[index] for item in sublist]  
    
    support_y = np.array(
        [img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
         for sublist in support_x_batch[index] for item in sublist]).astype(np.int32)
    
    flatten_query_x = [os.path.join(path, item)
                       for sublist in query_x_batch[index] for item in sublist]
    query_y = np.array([img2label[item[:9]]
                        for sublist in query_x_batch[index] for item in sublist]).astype(np.int32)
    
    unique = np.unique(support_y)
    random.shuffle(unique)
    
    support_y_relative = np.zeros(setsz)
    query_y_relative = np.zeros(querysz)
    
    for idx, l in enumerate(unique):
        support_y_relative[support_y == l] = idx
        query_y_relative[query_y == l] = idx
        
    support_y_relative = torch.LongTensor(support_y_relative)
    query_y_relative = torch.LongTensor(query_y_relative)
        
    for i, path in enumerate(flatten_support_x):
        support_x[i] = transform(path)

    for i, path in enumerate(flatten_query_x):
        query_x[i] = transform(path)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
