import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data.dataloader import default_collate
import torchvision.transforms.functional as TF


class kitti_dataset(Dataset):
    def __init__(self, root, annotation, depth, transform=None):
        self.root = root
        self.anno_root = annotation
        self.depth_root = depth
        self.transform = transform
        
        self.images = sorted(os.listdir(self.root))
        self.annotations = sorted(os.listdir(self.anno_root))
        self.depths = sorted(os.listdir(self.depth_root))

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.images[index])).convert('RGB')
        annotation = open(os.path.join(self.anno_root, self.annotations[index])).read().split('\n')
        depth = torch.load(os.path.join(self.depth_root, self.depths[index].split('.')[0] + '.pt'))
        if self.transform is not None:
            img = self.transform(image)

        original_x, original_y = depth.shape[-1], depth.shape[-2]   
        depth = TF.resize(depth, size=img.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR)        
        
        bboxes = []
        for anno in annotation:
            if anno == '':
                break
            else:
                anno = anno.split(' ')
                bbox = [
                    int(float(anno[4]) / original_x * 256), 
                    int(float(anno[5]) / original_y * 256), 
                    int((float(anno[6]) - float(anno[4])) / original_x * 256), 
                    int((float(anno[7]) - float(anno[5])) / original_y * 256)
                ]
                bboxes.append(bbox)
        
        return img, bboxes, depth

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    max_boxes = max(len(item[1]) for item in batch)
    
    batch_padded = []
    for data, boxes, depths in batch:
        padded_boxes = boxes + [[-1, -1, -1, -1] for _ in range(max_boxes - len(boxes))]
        padded_boxes = torch.tensor(padded_boxes)
        batch_padded.append((data, padded_boxes, depths))

    return default_collate(batch_padded)