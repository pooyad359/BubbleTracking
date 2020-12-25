# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Setup

# %cd ..

import json
import os
from pathlib import Path
from tqdm.auto import tqdm
from utils.fileio import download

path = Path('./data/masks/')

with open('./data/annotations/annotation.json','r') as fp:
    annot = json.load(fp)


def view_list(x):
    for i, o in enumerate(x):
        print(f'{i}.\t',o)


view_list(annot[0].keys())


def save_binary(content,filepath):
    with open(filepath,'wb') as fp:
        fp.write(content)


# # Extract info from json

# +
data = {}
for img_data in tqdm(annot,desc = 'Images'):
    img_id = os.path.splitext(img_data['External ID'])[0]
    labels = img_data['Label']
    for obj in labels.get('objects',[]):
        mask_id = obj['featureId']
        mask_uri = obj['instanceURI']
        row = [img_id,mask_id,mask_uri]
        data[f'{img_id}_{mask_id}.png'] = mask_uri

        
    
    
# -

len(data.items())

# # Download Masks

download(data,'./data/masks/',16)

# # Aggregate masks

img_list = []
for img_data in tqdm(annot,desc = 'Images'):
    img_id = img_data['External ID']
    img_list.append(img_id)

os.listdir(path)

import cv2
import matplotlib.pyplot as plt

q = cv2.imread(str(path/'frame000120_cki8va6d0160c0ycz3k5c7nkd.png'),cv2.IMREAD_GRAYSCALE)

q.shape

plt.imshow(q)

output_path = './data/joined_masks/'
size = (1080, 1920)
for img in tqdm(img_list):
    img_id = os.path.splitext(img)[0]
    files = [m for m in os.listdir(path) if m.startswith(img_id)]
    masks = [cv2.imread(str(path/m),cv2.IMREAD_GRAYSCALE) for m in files]
#     list_masks = np.stack(arrays,axis=2)
    
    final_mask = np.zeros(size)
    for i, mask in enumerate(masks):
        final_mask[mask>128] = i
    filepath = os.path.join(output_path,img_id+'.png')
    cv2.imwrite(filepath,final_mask)
    print(img_id,len(masks),final_mask.shape)
