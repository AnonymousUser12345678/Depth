import torch
from PIL import Image


@torch.no_grad()
def extract_iou(img):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.cuda()

    results = model(img)
    df = results.pandas().xyxy[0]

    return results, df