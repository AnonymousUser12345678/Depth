import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torchvision

from time import time

import numpy as np

import wandb
import argparse
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_table

from utils import extract_bbox, get_psnr, get_ssim


def train(student, train_loader, optimizer, criterion, epoch, args):
    student.train()
    losses = []
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit='batch') as pbar:
        for images, bboxes, depths in pbar:
            depths = depths.squeeze()
            if args.global_info:
                inputs, labels, global_infos = extract_bbox(images, bboxes, depths, args)
                global_infos = torch.stack(global_infos).cuda()
                global_infos = global_infos.view(global_infos.shape[0], -1)
            else:
                inputs, labels = extract_bbox(images, bboxes, depths, args)

            if len(inputs) == 0:
                continue
            else:
                inputs = torch.stack(inputs)
                labels = torch.stack(labels)

            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = student(inputs, global_infos if args.global_info else None)
            if student.loss == 'kd':
                labels = labels.round()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix(loss = np.mean(losses))
    wandb.log({"training loss": np.mean(losses)})


@torch.no_grad()
def test(student, val_loader, criterion, epoch, args):
    student.eval()
    losses = []
    psnrs = []
    ssims = []
    max_diff = []
    min_diff = []
    mean_diff = []
    
    for images, bboxes, depths in val_loader:
        depths = depths.squeeze()
        if args.global_info:
            inputs, labels, global_infos = extract_bbox(images, bboxes, depths, args)
            global_infos = torch.stack(global_infos).cuda()
            global_infos = global_infos.view(global_infos.shape[0], -1)
        else:
            inputs, labels = extract_bbox(images, bboxes, depths, args)
            global_infos = None
        
        if len(inputs) == 0:
            continue
        else:
            inputs = torch.stack(inputs)
            labels = torch.stack(labels)

        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = student(inputs, global_infos)

        max_ = torch.abs(
            outputs.view(outputs.shape[0], -1).max(1)[0] \
            - labels.view(labels.shape[0], -1).max(1)[0])
        min_ = torch.abs(
            outputs.view(outputs.shape[0], -1).min(1)[0] \
            - labels.view(labels.shape[0], -1).min(1)[0])
        mean_ = torch.abs(
            outputs.view(outputs.shape[0], -1).mean(1) \
            - labels.view(labels.shape[0], -1).mean(1))

        max_diff.append(max_.mean().item())
        min_diff.append(min_.mean().item())
        mean_diff.append(mean_.mean().item())

        for output, label in zip(outputs, labels):
            psnr = get_psnr(output, label)
            psnrs.append(psnr.item())
            ssim = get_ssim(output, label)
            ssims.append(ssim.item())

    t = time()
    for _ in range(100):
        if args.global_info:
            student(inputs[0].unsqueeze(0), global_infos[0].unsqueeze(0))
        else:   
            student(inputs[0].unsqueeze(0))
    inference_time = time() - t
    print(f"Inference (1-batch 100-times average): {inference_time/100}")
    print(f"PSNR: {np.mean(psnrs)}")
    print(f"SSIM: {np.mean(ssims)}")
    print(f"MAX_DIFF: {np.mean(max_diff)}")
    print(f"MIN_DIFF: {np.mean(min_diff)}")
    print(f"MEAN_DIFF: {np.mean(mean_diff)}")
    
    flops = FlopCountAnalysis(student, inputs[0].unsqueeze(0))
    print(f"FLOPS: {flops.total()}")

    wandb.log({"validation loss": np.mean(losses)})
    wandb.log({"validation psnr": np.mean(psnrs)})
    wandb.log({"validation ssim": np.mean(ssims)})
    wandb.log({"validation max_diff": np.mean(max_diff)})
    wandb.log({"validation min_diff": np.mean(min_diff)})
    wandb.log({"validation mean_diff": np.mean(mean_diff)})

    with open(args.save_txt, 'w') as file:
        file.write(f"Inference (1-batch 100-times average): {inference_time/100}\n")
        file.write(f"PSNR: {np.mean(psnrs)}\n")
        file.write(f"SSIM: {np.mean(ssims)}\n")
        file.write(f"MAX_DIFF: {np.mean(max_diff)}\n")
        file.write(f"MIN_DIFF: {np.mean(min_diff)}\n")
        file.write(f"MEAN_DIFF: {np.mean(mean_diff)}\n")
        # file.write(f"FLOPS: {flops}\n")