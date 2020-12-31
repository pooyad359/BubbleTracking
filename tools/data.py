import torch
import os
import numpy as np
from PIL import Image, ImageDraw


class BubbleDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, mask_path,scale = 1):
        super(BubbleDataset, self).__init__()
        image_id = os.listdir(mask_path)
        ids = []
        for iid in image_id:
            path = os.path.join(mask_path, iid)
            mask = Image.open(path) 
            if np.max(mask)>0:
                ids.append(iid)
        self.image_id = ids
        self.scale = scale
        self.image_path = img_path
        self.mask_path = mask_path

    def get_boxes(self, idx):
        mask, _ = self.get_mask(idx)
        array = np.array(mask)
        if array.max() == 0:
            return []

        obj_ids = np.unique(array)[1:]

        boxes = []
        scale = self.scale
        for i in obj_ids:
            pos = np.where(array == i)
            xmin = np.min(pos[1])*scale
            xmax = np.max(pos[1])*scale
            ymin = np.min(pos[0])*scale
            ymax = np.max(pos[0])*scale
            boxes.append([xmin, ymin, xmax, ymax])
        return np.array(boxes)

    def get_image(self, idx):
        path = os.path.join(self.image_path, self.image_id[idx])
        image = Image.open(path)
        if self.scale !=1:
            scale = self.scale
            size = (int(d*scale) for d in image.size)
            image = image.resize(size,Image.BICUBIC)
        return image

    def get_mask(self, idx):
        path = os.path.join(self.mask_path, self.image_id[idx])
        mask = Image.open(path)
        if self.scale!=1:
            scale = self.scale
            size = (int(d*scale) for d in mask.size)
            mask = mask.resize(size,Image.BICUBIC)
        if np.max(mask) == 0:
            return mask, np.array(mask)
        nmax = int(np.max(mask))
        
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
            target['masks'] = torch.as_tensor(masks[:len(usable)],dtype = torch.uint8)[usable]
            target['image_id'] = torch.tensor([idx])
            target['area'] = area[usable]
            target['iscrowd'] = torch.zeros((len(usable), ), dtype=torch.int64)

        return torch.tensor(img).permute(2, 0, 1).float(), target

    def __len__(self):
        return len(self.image_id)

    @property
    def length(self):
        return len(self)
