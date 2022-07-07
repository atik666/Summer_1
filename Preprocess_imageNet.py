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

root = '/home/admin1/Documents/Atik/Meta_Learning/MAML-Pytorch/datasets/miniImageNet/'

path = os.path.join(root, 'images')

dictLabels = loadCSV(os.path.join(root, mode + '.csv'))  # csv path