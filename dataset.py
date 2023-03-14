import os
import cv2
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, txt_file):
        self.image_path = []
        self.label = []
        f = open(txt_file, "r")
        lines = f.readlines()
        for line in lines:    
            a = line.split(' ') # assign to the variable a
            self.image_path.append(a[0]) 
            a[1] = a[1].strip('\n') # remove '\n'
            self.label.append(a[1])
        
    def __len__(self):
        return len(self.label)  

    def __getitem__(self, index):
        filename = self.image_path[index]
        label = self.label[index]
        image = self.read_img(filename)
        length = self.__len__()
        #print(length)
        return image, label

    def read_img(self, img):
        n = cv2.imread(img)
        n = torch.tensor(n)
        #segmentation
        n = n[0:200, 0:200]
        return n
'''
def onehot_encoding(label_list):
    encoded_label_list = torch.empty(0, 50)

    for label in label_list:
        tmp = torch.zeros(1, 50)
        tmp[:, int(label)] = 1
        encoded_label_list = torch.cat([encoded_label_list, tmp], dim=0)
    #encoded_label_list = torch.tensor(encoded_label_list)

    return encoded_label_list

def turn_to_score(score_vector):
    maximum = score_vector[0]
    idx = 0
    max_index = 0
    while idx < len(score_vector):
        if score_vector[idx] > maximum:
            maximum = score_vector[idx]
            max_index = idx
        idx = idx + 1
    return max_index
    
def accuracy(label, score_list):
    num = 0
    for idx in range(len(label)):
        if label[idx] == score_list[idx]:
            num = num + 1
    return num/len(label)
'''

if __name__ == "__main__":
    traindataset = CustomImageDataset("train.txt")
    print(traindataset.__len__())
    
    
