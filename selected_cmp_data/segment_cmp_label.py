import torchvision
import numpy
import torch
import segmentation_utils
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import glob
from datetime import datetime
import cv2

# set computation device
device = torch.device('cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

images = []
for img in glob.glob("train_images/*.png"):
    images.append(Image.open(img))

model = torchvision.models.segmentation.fcn_resnet101(num_classes=9).to(device)
model.load_state_dict(torch.load('saved_model.pt'))
model.eval()

n = 0
for image in images:
    n += 1
    print(n)
    # do forward pass and get the output dictionary
    outputs = model(transform(image).unsqueeze(0).to(device))
    # get the data from the `out` key
    outputs = outputs['out']

    segmented_image = segmentation_utils.draw_segmentation_map(outputs)
    final_image = segmentation_utils.image_overlay(image, segmented_image)
    cv2.imwrite(image.filename, final_image)
