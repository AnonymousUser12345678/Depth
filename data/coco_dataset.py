import os
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.functional as TF


class COCODataset(Dataset):
    def __init__(self, root, annotation, depth, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.depth_root = depth
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annotation = self.coco.loadAnns(ann_ids)
        
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        depth = torch.load(os.path.join(self.depth_root, path.split('.')[0] + '.pt'))

        if self.transform is not None:
            img = self.transform(img)
        
        original_x, original_y = depth.shape[-2], depth.shape[-1]   

        depth = TF.resize(depth, size=img.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR)

        bboxes = []
        for anno in coco_annotation:
            bbox = anno['bbox']
            bbox = [
                bbox[0] / original_y * 256, 
                bbox[1] / original_x * 256, 
                bbox[2] / original_y * 256, 
                bbox[3] / original_x * 256
            ]
            bboxes.append(bbox)
        return img, bboxes, depth

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    max_boxes = max(len(item[1]) for item in batch)
    
    batch_padded = []
    for data, boxes, depths in batch:
        padded_boxes = boxes + [[-1, -1, -1, -1] for _ in range(max_boxes - len(boxes))]
        padded_boxes = torch.tensor(padded_boxes)
        batch_padded.append((data, padded_boxes, depths))

    return default_collate(batch_padded)