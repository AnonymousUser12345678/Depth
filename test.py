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
sys.path.append('UniDepth')
import wandb
wandb.init(project='depth', entity='sungheoj')

from model.iou.iou import extract_iou
from UniDepth.teacher import load_teacher
from model.student import Student

from data.get_loader import get_loader

from train import train, test
from tqdm import tqdm
from utils import extract_bbox
import torchsummary


def main(args):
    print(args)
    train_loader, val_loader = get_loader(args)

    student = Student(args).cuda().eval()
    
    args.global_info = False
    student1 = Student(args).cuda().eval()
    student1.load_state_dict(torch.load(args.ckpt_path1))
    criterion = torch.nn.MSELoss()

    args.global_info = True
    student2 = Student(args).cuda().eval()
    student2.load_state_dict(torch.load(args.ckpt_path2))
    
    print(torchsummary.summary(
        student, 
        (3, args.student_input_size, args.student_input_size)))
    test(student, val_loader, criterion, 1, args)

    count = 0
    for idx, (images, bboxes, depths) in tqdm(enumerate(val_loader)):
        depths = depths.squeeze()

        inputs, labels, global_infos = extract_bbox(images, bboxes, depths, args)
        global_infos = torch.stack(global_infos).cuda()
        global_infos = global_infos.view(global_infos.shape[0], -1)
        
        if len(inputs) == 0:
            continue
        else:
            inputs = torch.stack(inputs)
            labels = torch.stack(labels)

        inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs1 = student1(inputs, None)
        outputs2 = student2(inputs, global_infos)
        # os.makedirs(os.path.join(args.result_path, f'{idx}'), exist_ok=True)
        
        for kk, boxes in enumerate(bboxes[0]):
            if boxes[0] != -1:
                img = Image.fromarray((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                draw = ImageDraw.Draw(img)
                
                x1, y1, width, height = boxes
                x1, width = x1 / 256 * images[0].shape[1], width / 256 * images[0].shape[2]
                y1, height = y1 / 256 * images[0].shape[1], height / 256 * images[0].shape[2]
                
                x1, y1, width, height = int(x1), int(y1), int(width), int(height)

                plt.figure(figsize=(16, 4))

                plt.subplot(1, 5, 1)
                rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                plt.imshow(img)
                # plt.title('Image')
                plt.axis('off')
                
                plt.subplot(1, 5, 2)
                rgb_img = TF.crop(images[0], y1, x1, height, width)
                rgb_img = TF.resize(rgb_img, (args.student_input_size, args.student_input_size))
                plt.imshow(rgb_img.permute(1,2,0).cpu().numpy())
                # plt.imshow(depths[0].squeeze().detach().cpu().numpy())
                # rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
                # plt.gca().add_patch(rect)
                # plt.title('Depth')
                plt.axis('off')

                plt.subplot(1, 5, 3)
                plt.imshow(labels[kk].squeeze().detach().cpu().numpy(), cmap='magma_r')
                # plt.title('Box Depth')
                plt.axis('off')
                
                plt.subplot(1, 5, 4)
                plt.imshow(outputs1[kk].squeeze().detach().cpu().numpy(), cmap='magma_r')
                plt.axis('off')
                
                plt.subplot(1, 5, 5)
                plt.imshow(outputs2[kk].squeeze().detach().cpu().numpy(), cmap='magma_r')
                # # plt.title('Box Prediction')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(args.result_path, f'bbox_{count}.png'))
                plt.clf()
                count += 1
        
                
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
    args.ckpt_path1 = f'checkpoints/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_False.pth'
    args.ckpt_path2 = f'checkpoints/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_True.pth'
    args.result_path = f'results/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}' 
    if args.global_info:
        args.ckpt_path = f'checkpoints/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_True.pth'
        args.result_path = f'results/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_global_info' 
    
    main(args)