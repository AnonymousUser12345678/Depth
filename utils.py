import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from torchvision.transforms import transforms
from skimage.metrics import structural_similarity as ssim


def get_psnr(img1, img2):
    img1_norm = img1 / torch.max(img1)
    img2_norm = img2 / torch.max(img2)
    
    mse = torch.mean((img1_norm - img2_norm)**2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_ssim(img1, img2):
    img2 = img2.squeeze().cpu().numpy()
    img1 = img1.squeeze().cpu().numpy()
    img2 = (img2 * 255).astype(np.uint8)
    img1 = (img1 * 255).astype(np.uint8)

    ssim_value = ssim(img2, img1)
    return ssim_value


def extract_and_mirror_diagonal_points(center, img_size, n):
    cx, cy = center
    max_x, max_y = img_size - 1, img_size - 1
    diagonal_points = [(cx + i, cy + i) for i in range(-n, n + 1) if (0 <= cx + i <= max_x and 0 <= cy + i <= max_y)]
    mirrored_points = [(cx - i, cy + i) for i in range(-n, n + 1) if (0 <= cx - i <= max_x and 0 <= cy + i <= max_y)]

    return diagonal_points, mirrored_points


def extract_pixel_values(image_tensor, points):
    values = [image_tensor[y, x] for x, y in points]
    return values


def resize_tensor_image(image_tensor, new_size):
    resize_transform = transforms.Resize(new_size)
    resized_image_tensor = resize_transform(image_tensor)
    return resized_image_tensor


def extract_bbox(images, bboxes, depths, args):
    inner_box_images = []
    inner_box_depths = []
    global_infos = []
    for image, bbox, depth in zip(images, bboxes, depths):
        for box in bbox:
            if box[0] != -1:
                x1, y1, width, height = box
                x1, y1, width, height = int(x1), int(y1), int(width), int(height)
                
                iou_img = image[:, y1:y1+height, x1:x1+width]
                iou_depth = depth[y1:y1+height, x1:x1+width]
                
                if iou_img.shape[-2] < 10 or iou_img.shape[-1] < 10:
                    continue
                
                if args.global_info:
                    center = (x1+int(width/2), y1+int(height/2))
                    
                    global_info1 = depth[:, x1+int(width/2)]
                    global_info2 = depth[y1+int(height/2), :]
                    
                    global_info3, global_info4 = extract_and_mirror_diagonal_points(
                        center, depth.shape[-1], 256)
                    global_info3 = extract_pixel_values(depth, global_info3)
                    global_info3 = torch.stack(global_info3, -1).unsqueeze(0)
                    global_info3 = [F.interpolate(
                        t.unsqueeze(0).unsqueeze(0), size=256, mode='linear', align_corners=True).squeeze() for t in global_info3]
                    global_info3 = torch.stack(global_info3)

                    global_info4 = extract_pixel_values(depth, global_info4)
                    global_info4 = torch.stack(global_info4, -1).unsqueeze(0)
                    global_info4 = [F.interpolate(
                        t.unsqueeze(0).unsqueeze(0), size=256, mode='linear', align_corners=True).squeeze() for t in global_info4]
                    global_info4 = torch.stack(global_info4)
                    
                    global_info = torch.cat(
                        [global_info1.unsqueeze(0), global_info2.unsqueeze(0), global_info3, global_info4], dim=0).squeeze()
                    global_infos.append(global_info)
                
                iou_img = TF.resize(
                    iou_img, 
                    (args.student_input_size, args.student_input_size), 
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                iou_depth = TF.resize(
                    iou_depth.unsqueeze(0), 
                    (args.student_input_size, args.student_input_size), 
                    interpolation=TF.InterpolationMode.BILINEAR
                )
                inner_box_images.append(iou_img)
                inner_box_depths.append(iou_depth)
            
    if args.global_info:
        return inner_box_images, inner_box_depths, global_infos
    else:
        return inner_box_images, inner_box_depths

