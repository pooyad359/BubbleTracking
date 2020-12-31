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

# %load_ext autoreload
# %autoreload 2

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import numpy as np
import PIL
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from utils import collate_fn
from tools.data import BubbleDataset

# # Create Datasets

imgs_path = Path('./data/images/Source/')
mask_path_train = Path('./data/joined_masks/train/')
mask_path_test = Path('./data/joined_masks/test/')

dstr = BubbleDataset(imgs_path,mask_path_train,scale=.5)
dsval =BubbleDataset(imgs_path,mask_path_test,scale=.5)

loader = DataLoader(
    dstr,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
test_loader = DataLoader(
    dsval,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

# # Create Model

import utils
from engine import train_one_epoch, evaluate
from models.mask_rcnn import MaskRCNN

num_classes = 2
model = MaskRCNN(num_classes)

# # Train Model

device =torch.device('cuda')
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr = 0.001, weight_decay=1e-3)
lr_sch = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.1)

epochs = 1
for epoch in range(epochs):
    train_one_epoch(model,optimizer,loader,device,epoch,12)
    lr_sch.step()
    evaluate(model,test_loader,device=device)

# # Evaluation

evaluate(model,test_loader,device=device)

# # Testing

model = model.to('cpu')
idx = 20
img, _ = dsval[idx]
with torch.no_grad():
    preds = model([img])

preds


# +
def show_tensor(t):
    return Image.fromarray(t.permute(1,2,0).numpy().astype(np.uint8))

def show_results(img,preds):
    image = show_tensor(img)
    boxes = preds['boxes']
    scores = preds['scores']
    draw = ImageDraw.Draw(image)
    for b in boxes:
        draw.rectangle(list(b.detach().numpy()), outline='red', width=3)
    return image


# -

show_results(img,preds[0])

# # Save the model

import pickle

with open('model.pkl','wb') as fp:
    pickle.dump(model.state_dict(),fp)
