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

# # Create Datasets

imgs_path = Path('./data/images/Source/')
mask_path_train = Path('./data/joined_masks/train/')
mask_path_test = Path('./data/joined_masks/test/')


class BubbleDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, mask_path):
        super(BubbleDataset, self).__init__()
        image_id = os.listdir(mask_path)
        ids = []
        for iid in image_id:
            path = os.path.join(mask_path, iid)
            mask = Image.open(path) 
            if np.max(mask)>0:
                ids.append(iid)
        self.image_id = ids
        self.image_path = img_path
        self.mask_path = mask_path

    def get_boxes(self, idx):
        mask, _ = self.get_mask(idx)
        array = np.array(mask)
        if array.max() == 0:
            return []

        obj_ids = np.unique(array)[1:]

        boxes = []
        for i in obj_ids:
            pos = np.where(array == i)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        return np.array(boxes)

    def get_image(self, idx):
        path = os.path.join(self.image_path, self.image_id[idx])
        image = Image.open(path)
        return image

    def get_mask(self, idx):
        path = os.path.join(self.mask_path, self.image_id[idx])
        mask = Image.open(path)
        if np.max(mask) == 0:
            return mask, np.array(mask)
        nmax = np.max(mask)
        layered = np.zeros((nmax, np.shape(mask)[0], np.shape(mask)[1]))
        for i in range(nmax):
            layered[i, :, :] = mask == (i + 1)
        return mask, layered

    def visualise(self, idx):
        img = self.get_image(idx)
        boxes = self.get_boxes(idx)
        draw = ImageDraw.Draw(img)
        for i in range(boxes.shape[0]):
            draw.rectangle(list(boxes[i, :]), outline='red', width=3)
        return img

    def __getitem__(self, idx):
        target = {}
        boxes = self.get_boxes(idx)
        _, masks = self.get_mask(idx)
        if len(boxes) == 0:
            area = torch.tensor([],dtype=torch.float)
            img = self.get_image(idx)
            img = np.array(img)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float)
            target['labels'] = torch.ones((0, ), dtype=torch.int64)
            target['masks'] = torch.as_tensor(masks,dtype = torch.uint8)
            target['image_id'] = torch.tensor([idx])
            target['area'] = area
            target['iscrowd'] = torch.zeros((0, ), dtype=torch.int64)
        else:
            area = torch.as_tensor(
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                dtype=torch.float)
            usable = area>0
            _, masks = self.get_mask(idx)
            img = self.get_image(idx)
            img = np.array(img)
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)[usable]
            target['labels'] = torch.ones((len(usable), ), dtype=torch.int64)
            target['masks'] = torch.as_tensor(masks,dtype = torch.uint8)[usable]
            target['image_id'] = torch.tensor([idx])
            target['area'] = area[usable]
            target['iscrowd'] = torch.zeros((len(usable), ), dtype=torch.int64)

        return torch.tensor(img).permute(2, 0, 1).float(), target

    def __len__(self):
        return len(self.image_id)

    @property
    def length(self):
        return len(self)


dstr = BubbleDataset(imgs_path,mask_path_train)
dsval =BubbleDataset(imgs_path,mask_path_test)

loader = DataLoader(dstr,
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

# +
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import utils
from utils import collate_fn
from engine import train_one_epoch, evaluate


# +
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# -

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


num_classes = 2
model = get_model_instance_segmentation(num_classes)

# # Train Model

device =torch.device('cuda')
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr = 0.001, weight_decay=1e-3)
lr_sch = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.1)

epochs = 10
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
