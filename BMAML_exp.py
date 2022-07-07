import os
from os import walk
import glob
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

root = '/home/admin1/Documents/Atik/Meta_Learning/MAML-Pytorch/datasets/256'

path = os.path.join(root, 'train/') 

filenames = next(walk(path))[1]

dictLabels = {}

for i in range(len(filenames)):  
    img = []
    for images in glob.iglob(f'{path+filenames[i]}/*'):
        # check if the image ends with png
        if (images.endswith(".jpg")):
            img_temp = images[len(path+filenames[i]+'/'):]
            img_temp = filenames[i]+'/'+img_temp
            img.append(img_temp)
        
        dictLabels[filenames[i]] = img

# resize = 84
# transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
#                                      transforms.Resize((resize, resize)),
#                                      # transforms.RandomHorizontalFlip(),
#                                      # transforms.RandomRotation(5),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#                                      ])

#file = next(iter(dictLabels.items()))[1]

file = list(dictLabels.values())
# img = transform(path+file[0])

# img = np.asarray(img)
# img = np.moveaxis(img, 2, 0)
# img_int = img.astype(np.uint8)


import cv2
import random

class_num = 0
num_samples = 10

random.shuffle(file[class_num])
samples = file[class_num][0:num_samples]

image_array = []
for i in range(num_samples):
    rand_class = int(np.random.randint(256,size=(1, 1)))
    rand_image = int(np.random.randint(100,size=(1, 1)))
    
    while rand_class == class_num:
        rand_class = int(np.random.randint(256,size=(1, 1)))
    
    file_array = file[rand_class][rand_image]
    image_array.append(file_array)
    


img = [cv2.imread(path+samples[i]) for i in range(num_samples)]
img_neg = [cv2.imread(path+image_array[i]) for i in range(num_samples)]

img2 = [np.expand_dims(img[i].mean(axis=2).flatten(), axis=1) for i in range(len(img))]
img_neg = [np.expand_dims(img_neg[i].mean(axis=2).flatten(), axis=1) for i in range(len(img_neg))]

def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

img2 = [normalizeData(img2[i]) for i in range(len(img2))]
img_neg = [normalizeData(img_neg[i]) for i in range(len(img_neg))]

def orb_sim(img1, img2):
  # SIFT is no longer available in cv2 so using ORB
  orb = cv2.ORB_create()

  # detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)

  # define the bruteforce matcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  #perform matches. 
  matches = bf.match(desc_a, desc_b)
  #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
  similar_regions = [i for i in matches if i.distance < 50]  
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


""""""

img = cv2.imread(path+file[0])
vals = img.mean(axis=2).flatten()

# import matplotlib.pyplot as plt
# from sklearn import preprocessing

# vals = img.mean(axis=2).flatten()
# # plot histogram with 255 bins
# b, bins, patches = plt.hist(vals, 255)
# b = np.expand_dims(b, 1)
# b = preprocessing.normalize(b)
# plt.xlim([0,255])
# plt.show()


# img2 = cv2.imread(path+file[2])


# import matplotlib.pyplot as plt
# vals2 = img2.mean(axis=2).flatten()
# # plot histogram with 255 bins
# b2, bins, patches = plt.hist(vals2, 255)
# b2 = np.expand_dims(b2, 1)
# b2 = preprocessing.normalize(b2)
# plt.xlim([0,255])
# plt.show()

# dist = np.linalg.norm(b - b2)

# from scipy import signal

# d1, d2, d3 = img.shape

# x_data_reshaped = img.reshape((d1*d2*d3))
# x_data_reshaped2 = img2.reshape((d1*d2*d3))

# corr = signal.correlate(x_data_reshaped, x_data_reshaped2)




