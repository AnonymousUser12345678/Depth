import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import argparse
import sys
import os
from time import time
sys.path.append('UniDepth')
# import wandb
# wandb.init(project='depth', entity='sungheoj')

from model.iou.iou import extract_iou
from UniDepth.teacher import load_teacher
from model.student import Student

from data.get_loader import get_loader

from train import train, test
from tqdm import tqdm
from utils import extract_bbox
import torchsummary
import warnings
warnings.filterwarnings("ignore")


def main(args):
    train_loader, val_loader = get_loader(args)
    
    student = Student(args).cuda().eval()

    inputs = torch.randn(1, 3, args.student_input_size, args.student_input_size).cuda()
    global_infos = torch.randn(1, 1024).cuda()
    t = time()
    for i in range(10000):
        outputs = student(inputs, global_infos if args.global_info else None)
    print((time() - t) / 10000)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="coco", choices=['coco', 'kitti'], help="dataset name")
    
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--backbone", type=str, default="vitl14", choices=["vitl14", "cnvnxtl"])

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "kd"], help="loss function")
    parser.add_argument("--ckpt_path", type=str, default='checkpoints/coco_mse_resnet18_64.pth')
    
    parser.add_argument("--student_arch", type=str, default='resnet18', choices=['resnet18', 'resnet34'], help="student architecture")
    parser.add_argument("--student_input_size", type=int, default=64, choices=[32, 64], help="student input size")
    parser.add_argument("--global_info", action='store_true', default=False)
    
    args = parser.parse_args()
    main(args)