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
from models.mask_rcnn import MaskRCNN
from tools.data import BubbleDataset
import utils
from engine import train_one_epoch, evaluate
import argparse
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--device','-d',type = str, default = 'cuda', help = 'Device used for training. (cuda or cpu)')
parser.add_argument('--scale', '-s', type = float, default = 1, help = 'Can be used to scale down the images for quicker training')
parser.add_argument('--batch-size','-bs', type = int, default = 1, help = 'Batch size for training')

parser.add_argument('--train-images', '-tri',type = str, default = './data/images/Source/', help = 'Path to training images')
parser.add_argument('--train-masks', '-trm',type = str, default = './data/joined_masks/train/', help = 'Path to training masks')
parser.add_argument('--test-images', '-tsi',type = str, default = './data/images/Source/', help = 'Path to test images')
parser.add_argument('--test-masks', '-tsm', type = str, default = './data/joined_masks/test/', help = 'Path to test masks')
parser.add_argument('--num-classes', '-n', type = int, default = 2, help = 'Number of classes (including background)')
parser.add_argument('--epochs','-e', type = int, default = 5, help = 'Number of training epochs.')
parser.add_argument('--checkpoint-path','-cp', type = str, default = '', help = 'Path to directory where checkpoints will be saved. If empty no checkpoint will be saved.')

def save_model(model,path,name = None):
    if name is None:
        filepath = os.path.join(path,uuid.uuid4().hex+'.pkl')
    else:
        filepath = os.path.join(path,name +'.pkl')
    with open(filepath,'wb') as fp:
        pickle.dump(model.state_dict(),fp)

if __name__ == '__main__':
    args = parser.parse_args()
    run_id = uuid.uuid4().hex
    print(f'\t* Starting Run {run_id}',flush = True)
    device = torch.device(args.device)
    bs = args.batch_size
    scale = args.scale
    num_classes = args.num_classes
    checkpoint = args.checkpoint_path
    
    print('\t* Creating datasets',flush = True)
    dstr = BubbleDataset(args.train_images,args.train_masks,scale=scale)
    dsval =BubbleDataset(args.test_images,args.test_masks,scale=scale)
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
    print('\t* Creating model object',flush = True)
    model = MaskRCNN(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr = 0.001, weight_decay=1e-3)
    lr_sch = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=.1)
    print('\t* Initiating training process',flush = True)
    for epoch in range(args.epochs):
        print(f'*** Epoch {epoch:03.0f} ***',flush = True)
        train_one_epoch(model,optimizer,loader,device,epoch,12)
        lr_sch.step()
        if len(checkpoint)>0:
            print(f'Saving model after epoch {epoch}',flush = True)
            save_model(model,path,f'{run_id}_{epoch:03.0f}')
        evaluate(model,test_loader,device=device)