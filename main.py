import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR
import wandb

import numpy as np
from PIL import Image
import argparse
import sys
sys.path.append('UniDepth')

from model.iou.iou import extract_iou
from UniDepth.teacher import load_teacher
from model.student import Student

from data.get_loader import get_loader

from train import train, test


def main(args):
    wandb_name = f"{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_{args.global_info}"
    wandb.init(project='depth', entity='sungheoj', name=wandb_name)
    print(args)
    # data loader
    # img = Image.open('demo/image.jpeg')    
    # iou_img, ious = extract_iou(img)
    train_loader, val_loader = get_loader(args)
    #---------------------------------------------------------
    # teacher depth
    # input = (3, width, height)
    # teacher = load_teacher(args) 
    # rgb = torch.from_numpy(np.array(Image.open('demo/image.jpeg'))).permute(2, 0, 1) # C, H, W
    # predictions = teacher.infer(rgb)
    # depth = predictions["depth"]
    #---------------------------------------------------------
    # student depth
    student = Student(args).cuda()
    # training
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.01)
    criterion = torch.nn.MSELoss()

    # train    
    for epoch in range(args.epochs):
        train(student, train_loader, optimizer, criterion, epoch, args)
        test(student, val_loader, criterion, epoch, args)        
        scheduler.step()
        torch.save(
            student.state_dict(), 
            f"checkpoints/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_{args.global_info}.pth"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="coco", choices=["coco", "kitti"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "kd"])
    
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

    parser.add_argument("--backbone", type=str, default="vitl14", choices=["vitl14", "cnvnxtl"])

    parser.add_argument("--student_input_size", type=int, default=64, choices=[32, 64], help="student input size")
    parser.add_argument("--student_arch", type=str, default='resnet18', choices=['resnet18', 'resnet34'])
    parser.add_argument("--global_info", action='store_true', default=False)
    
    args = parser.parse_args()

    args.save_txt = f"./checkpoints/{args.data}_{args.loss}_{args.student_arch}_{args.student_input_size}_{args.global_info}.txt"
    main(args)