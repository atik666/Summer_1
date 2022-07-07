import os
from os import walk
import glob
import numpy as np
import cv2
import random

root = '/home/admin1/Documents/Atik/Meta_Learning/MAML-Pytorch/datasets/256'

# path = os.path.join(root, 'train/') 

# filenames = next(walk(path))[1]

# dictLabels = {}

# for i in range(len(filenames)):  
#     img = []
#     for images in glob.iglob(f'{path+filenames[i]}/*'):
#         # check if the image ends with png
#         if (images.endswith(".jpg")):
#             img_temp = images[len(path+filenames[i]+'/'):]
#             img_temp = filenames[i]+'/'+img_temp
#             img.append(img_temp)
        
#         dictLabels[filenames[i]] = img

# file = list(dictLabels.values())

def load_files(loc: str) -> list:
    
    global mood
    path = os.path.join(root, mood, '')
    
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

    return list(dictLabels.values())
    
mood = 'train'
file = load_files(root)      

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

# def pos_class(classes: list, batch: int) -> list:
#     class_num = len(classes)
#     img_per_class = int(len(classes)[0]/batch)
    
    

    
#     return 

classes = file
batch = 10
class_num = len(classes)
batch_size = int(len(classes[0])/batch)

images = []
class_files = []
for k in range(class_num):

    classes_n = [classes[k][i:i+batch_size] \
                 for i in range(0, len(classes[k]), batch_size)]
    
    images_n = [[cv2.imread(os.path.join(root, mood, '')+classes_n[j][i]) \
                for i in range(len(classes_n[j]))] for j in range(len(classes_n))]
        
    images.append(images_n)
    class_files.append(classes_n)
    
    
random.shuffle(file)
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

def orb_sim(img1, img2):
    
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

similarity = [[orb_sim(img[j], img_neg[i]) for i in range(num_samples)] for j in range(num_samples)]

sim_val = np.mean([sum(similarity[i])/num_samples for i in range(num_samples)])


