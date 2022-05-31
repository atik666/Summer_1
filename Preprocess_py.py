import warnings
warnings.filterwarnings("ignore")
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import numpy as np
import os
import zipfile

root_path = '/home/admin1/Documents/Atik/Meta_Learning/MAML-Pytorch/datasets'
processed_folder =  os.path.join(root_path)

zip_ref = zipfile.ZipFile(os.path.join(root_path,'omniglot_standard.zip'), 'r')
zip_ref.extractall(root_path)
zip_ref.close()

root_dir = '/home/admin1/Documents/Atik/Meta_Learning/MAML-Pytorch/datasets/python'

import torchvision.transforms as transforms
from PIL import Image

'''
an example of img_items:
( '0709_17.png',
  'Alphabet_of_the_Magi/character01',
  './../datasets/omniglot/python/images_background/Alphabet_of_the_Magi/character01')
'''
def find_classes(root_dir):
    img_items = []
    for (root, dirs, files) in os.walk(root_dir): 
        for file in files:
            if (file.endswith("png")):
                r = root.split('/')
                img_items.append((file, r[-2] + "/" + r[-1], root))
    print("== Found %d items " % len(img_items))
    return img_items

def index_classes(items):
    class_idx = {}
    count = 0
    for item in items:
        if item[1] not in class_idx:
            class_idx[item[1]] = count
            count += 1
    print('== Found {} classes'.format(len(class_idx)))
    return class_idx

img_items =  find_classes(root_dir)
class_idx = index_classes(img_items)

temp = dict()
for imgname, classes, dirs in img_items:
    img = '{}/{}'.format(dirs, imgname)
    label = class_idx[classes]
    transform = transforms.Compose([lambda img: Image.open(img).convert('L'),
                              lambda img: img.resize((28,28)),
                              lambda img: np.reshape(img, (28,28,1)),
                              lambda img: np.transpose(img, [2,0,1]),
                              lambda img: img/255.
                              ])
    img = transform(img)
    if label in temp.keys():
        temp[label].append(img)
    else:
        temp[label] = [img]
print('begin to generate omniglot.npy')

img_list = []
for label, imgs in temp.items():
    img_list.append(np.array(imgs))
img_list = np.array(img_list).astype(np.float) # [[20 imgs],..., 1623 classes in total]
print('data shape:{}'.format(img_list.shape)) # (1623, 20, 1, 28, 28)
temp = []
np.save(os.path.join(root_dir, 'omniglot.npy'), img_list)
print('end.')