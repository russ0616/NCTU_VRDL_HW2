#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:15:07 2020

@author: user
"""

import numpy as np
import cv2
import os
import pandas as pd
import h5py
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import glob

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs

train_folder = "./data/train"
test_folder = "./data/test"
mat_file_name = 'digitStruct.mat'
mat_file = os.path.join(train_folder,mat_file_name)

f = h5py.File(mat_file,'r') 
all_rows = []
print('image bounding box data construction starting...')
bbox_df = pd.DataFrame([],columns=['height','img_name','label','left','top','width'])
for j in range(f['/digitStruct/bbox'].shape[0]):
    img_name = get_name(j, f)
    row_dict = get_bbox(j, f)
    row_dict['img_name'] = img_name
    all_rows.append(row_dict)
    bbox_df = pd.concat([bbox_df,pd.DataFrame.from_dict(row_dict,orient = 'columns')])
bbox_df['bottom'] = bbox_df['top']+bbox_df['height']
bbox_df['right'] = bbox_df['left']+bbox_df['width']
print('finished image bounding box data construction...')
bbox_df.to_csv('./train_data.csv',index = False)

# name = np.array(bbox_df['img_name'])
# first_label = (bbox_df[name == '1.png']).drop(['img_name'],axis=1)
# column_titles = ['label','height','width','top','left','bottom','right']
# first_label = np.array(first_label[column_titles])


data_list = pd.read_csv('./train_data.csv')
# data_list = data_list.set_index('Unnamed: 0')
# data_list = bbox_df
width=[]
height=[]
for i in range(len(data_list)):
    a = Image.open(os.path.join(train_folder,data_list['img_name'][i]))
    width.append(a.size[0])
    height.append(a.size[1])
data_list['img_width']=width
data_list['img_height']=height

## do normalize
new_list = data_list
new_list['top']=new_list['top']/new_list['img_height']
new_list['left']=new_list['left']/new_list['img_width']
new_list['bottom']=new_list['bottom']/new_list['img_height']
new_list['right']=new_list['right']/new_list['img_width']

new_list['width']=new_list['width']/new_list['img_width']
new_list['height']=new_list['height']/new_list['img_height']

## add center
new_list['x_center']= (new_list['left']+new_list['right'])/2   
new_list['y_center']= (new_list['top']+new_list['bottom'])/2   

## write txt
img_path = "./data/images/*.png"
img_num = glob.glob(img_path)
img_num = sorted(img_num)
name = np.array(new_list['img_name'])
for i in range(len(img_num)):
    pre_label = (data_list[name == str(i+1)+'.png'])
    pre_label = pre_label.reset_index()
    file_obj = open("./data/custom/labels/"+str(i+1)+".txt",'w')
    
    for j in range(pre_label.shape[0]):
        if pre_label['label'][j]==10:
            pre_label['label'][j]=0
        file_obj.write(str(pre_label['label'][j])+' '+str(pre_label['x_center'][j])+' '+str(pre_label['y_center'][j])
                       +' '+str(pre_label['width'][j])+' '+str(pre_label['height'][j])+'\n')
    file_obj.close()
    
    
file_obj = open("./data/custom/val.txt",'w')       
for i in range(23402,33402):
    file_obj.write('data/custom/images/'+str(i+1)+'.png')
    if i != 33401:
        file_obj.write('\n')
file_obj.close()

file_obj = open("./data/custom/train.txt",'w')       
for i in range(23402):
    file_obj.write('data/custom/images/'+str(i+1)+'.png')
    if i != 23401:
        file_obj.write('\n')
file_obj.close()
    
