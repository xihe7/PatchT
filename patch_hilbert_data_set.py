# * coding：utf-8 *
# 作者:Little Bear
# 创建时间：2020/2/7 17:47
import cv2
import sys
import numpy as np
import os
import glob
import time
from skimage import io, transform
from hilbert import hilbertCurve

class_dict = {'fake': 1, 'real': 0}

train_path = "/data0/RF/lstmtry/train/"

order2 = hilbertCurve(2)
order2 = np.reshape(order2,(16))
seq = np.linspace(0,15,16).astype(int)
hilbert_ind = np.lexsort((seq,order2))
def read_image1(img):
    need_number = 16
    images = []
    image = []
    cap = cv2.resize(img,(256,256))
    w=cap.shape[1]
    h=cap.shape[0]
    #print("w",w,"h",h)
    #frames_num = cap.get(7) - 1
    #step = frames_num // need_number
    #print("frames_num : ", frames_num, "step:", step)
    patch_size=(4, 4)
    patch_height, patch_width = int(h / patch_size[0]), int(w / patch_size[1])
    
    for i in range(4):
        for j in range(4):
            frames = cap[i*patch_height:(i+1)*patch_height,j*patch_width:(j+1)*patch_width,] 
            images.append(frames)
    for i in range(16):
        #print(hilbert_ind[i])
        image.append(images[hilbert_ind[i]])
    #print(image)
    return np.asarray(image, np.float32)
def read_image(image_path):
    need_number = 16
    images = []
    image = []
    cap = cv2.imread(image_path)
    cap = cv2.resize(cap,(256,256))
    w=cap.shape[1]
    h=cap.shape[0]
    #print("w",w,"h",h)
    #frames_num = cap.get(7) - 1
    #step = frames_num // need_number
    #print("frames_num : ", frames_num, "step:", step)
    patch_size=(4, 4)
    patch_height, patch_width = int(h / patch_size[0]), int(w / patch_size[1])
    
    for i in range(4):
        for j in range(4):
            frames = cap[i*patch_height:(i+1)*patch_height,j*patch_width:(j+1)*patch_width,] 
            images.append(frames)
    for i in range(16):
        #print(hilbert_ind[i])
        image.append(images[hilbert_ind[i]])
    #print(image)
    return np.asarray(image, np.float32)


def train_data(train_path):
    train = []
    label = []
    cate = [train_path + x for x in os.listdir(train_path) if os.path.isdir(train_path + x)]
    #print(cate)
    for index, folder in enumerate(cate):
        # glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表
        for path in glob.glob(folder + '/*.png') or glob.glob(folder + '/*.jpg'):
            #print("image read", path)
            imagepatch = read_image(path)
            imagepatch = imagepatch / 255
            print("image_shape:",imagepatch.shape)
            train.append(imagepatch)#此处的train是五个维度的(video_number,frame_number,320,160,3)
            label.append(index)
    return train, label


