import torch
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import sys
sys.path.append('UniDepth')


from model.iou.iou import extract_iou
from UniDepth.teacher import load_teacher
from model.student import Student
import torchsummary

import os
import matplotlib.pyplot as plt


data_path = {
    'coco_train_image': '/home/evan/dataset/coco/images/train2017',
    'coco_val_image': '/home/evan/dataset/coco/images/val2017',
    'coco_train_annotation': '/home/evan/dataset/coco/annotations/instances_train2017.json',
    'coco_val_annotation': '/home/evan/dataset/coco/annotations/instances_val2017.json',
    'kitti_image': '/home/jeong/dataset/KITTI/training/image_2',
    'kitti_annotation': '/home/jeong/dataset/KITTI/training/label_2',
}

save_path = {
    'coco_train_depth': '/home/evan/dataset/coco/images/depth_train2017',
    'coco_val_depth': '/home/evan/dataset/coco/images/depth_val2017',
    'kitti_train_depth': '/home/jeong/dataset/KITTI/training/depth_train',
}


def main(args):
    teacher = load_teacher(args) 
    if args.data == 'coco':
        for path in tqdm(os.listdir(data_path['coco_train_image'])):
            img_path = os.path.join(data_path['coco_train_image'], path)
            rgb = torch.from_numpy(np.array(Image.open(img_path)))
            if len(rgb.shape) == 2:
                rgb = rgb.unsqueeze(2).expand(-1, -1, 3)
            rgb = rgb.permute(2, 0, 1)
            predictions = teacher.infer(rgb)
            depth = predictions["depth"]

            torch.save(depth.cpu(), os.path.join(save_path['coco_train_depth'], path.split('.')[0] + '.pt'))

        for path in tqdm(os.listdir(data_path['coco_val_image'])):
            img_path = os.path.join(data_path['coco_val_image'], path)
            rgb = torch.from_numpy(np.array(Image.open(img_path)))
            if len(rgb.shape) == 2:
                rgb = rgb.unsqueeze(2).expand(-1, -1, 3)
            rgb = rgb.permute(2, 0, 1)
            
            predictions = teacher.infer(rgb)
            depth = predictions["depth"]
            torch.save(depth.cpu(), os.path.join(save_path['coco_val_depth'], path.split('.')[0] + '.pt'))

    elif args.data == 'kitti':
        for path in tqdm(os.listdir(data_path['kitti_image'])):
            img_path = os.path.join(data_path['kitti_image'], path)
            rgb = torch.from_numpy(np.array(Image.open(img_path)))
            if len(rgb.shape) == 2:
                rgb = rgb.unsqueeze(2).expand(-1, -1, 3)
            rgb = rgb.permute(2, 0, 1)
            predictions = teacher.infer(rgb)

            from time import time
            t = time()
            for i in range(100):
                predictions = teacher.infer(rgb)
            inference_time = time() - t
            print(f"Inference time: {inference_time}")
            import pdb; pdb.set_trace()


            depth = predictions["depth"]
            torch.save(depth.cpu(), os.path.join(save_path['kitti_train_depth'], path.split('.')[0] + '.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="coco", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")

    parser.add_argument("--backbone", type=str, default="vitl14", choices=["vitl14", "cnvnxtl"])
    
    args = parser.parse_args()
    main(args)