import cv2
import numpy as np
import torch
from label_color_map import label_color_map as label_map


def get_segment_labels(image, model, device):
    outputs = model(image)
    return outputs


def calculate_IoU(output, mask):
    # calculate the IoU for an image
    labels = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    n_inter = 0
    n_union = 0
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] == mask[i][j]:
                n_inter += 1
                n_union += 1
            else:
                n_union += 1
    return n_inter/n_union


def draw_segmentation_map(outputs):
    # create the RGB version of labels for model outputs
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image


def draw_RGB_labels(labels):
    # create the RGB version of labels for grayscale labels
    labels = np.array(labels)
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image


def image_overlay(image, segmented_image):
    # overlay the images with predicted labels
    alpha = 0.6  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image
