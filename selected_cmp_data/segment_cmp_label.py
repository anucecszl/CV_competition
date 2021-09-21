import torchvision
import torch
import segmentation_utils
from PIL import Image
import torchvision.transforms as transforms
import glob
import cv2

# set computation device
device = torch.device('cpu')
# create a transform function for data normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# load the images and labels to the computation device
# it's noted that the images should have same dimension with the labels.
images = []
for img in glob.glob("train_images/*.png"):
    images.append(Image.open(img))
# load the saved model and set it in evaluation mode
model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(num_classes=9).to(device)
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
    # save the overlaid images with predicted labels
    segmented_image = segmentation_utils.draw_segmentation_map(outputs)
    final_image = segmentation_utils.image_overlay(image, segmented_image)
    fp = 'segmentation_results/' + image.filename.split('\\')[1]
    cv2.imwrite(fp, final_image)
