import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import random_split

from data.coco_dataset import COCODataset, collate_fn
from data.kitti_dataset import kitti_dataset


data_path = {
    'coco_train_image': '/home/jeong/dataset/coco/images/train2017',
    'coco_val_image': '/home/jeong/dataset/coco/images/val2017',
    'coco_train_annotation': '/home/jeong/dataset/coco/annotations/instances_train2017.json',
    'coco_val_annotation': '/home/jeong/dataset/coco/annotations/instances_val2017.json',
    'coco_train_depth': '/home/jeong/dataset/coco/images/depth_train2017',
    'coco_val_depth': '/home/jeong/dataset/coco/images/depth_val2017',
    'kitti_image': '/home/jeong/dataset/KITTI/training/image_2',
    'kitti_annotation': '/home/jeong/dataset/KITTI/training/label_2',
    'kitti_depth': '/home/jeong/dataset/KITTI/training/depth_train',
}


def get_loader(args):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
        
    if args.data == 'coco':
        train_dataset = COCODataset(
            data_path['coco_train_image'], 
            data_path['coco_train_annotation'], 
            data_path['coco_train_depth'],
            transform=transform)
        val_dataset = COCODataset(
            data_path['coco_val_image'],
            data_path['coco_val_annotation'],
            data_path['coco_val_depth'],
            transform=transform)
    else:
        train_val_dataset = kitti_dataset(
            data_path['kitti_image'], 
            data_path['kitti_annotation'], 
            data_path['kitti_depth'],
            transform=transform)

        torch.manual_seed(42)
        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_val_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True,
        collate_fn=collate_fn)
    
    return train_loader, val_loader