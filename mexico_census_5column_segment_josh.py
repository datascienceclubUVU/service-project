# Need to set up out dir and also how it is being split. Bounding boxes?

# Some basic setup
# import some common libraries
import csv
import os
import random
import statistics
import subprocess
import traceback
from glob import glob
from io import BytesIO
from shlex import quote
from sys import argv

import cv2
import detectron2
import matplotlib.pyplot as plt
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode  # I added this
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from google.cloud import storage
from PIL import Image

# Setup detectron2 logger
setup_logger()

numdir = argv[1]
album = argv[2]


# Set Up Models
# the cfg object here is an instantiation of the model. The merge_from_file function gets arguments from a default YAML
# file to configure the model. The functions that follow update certain arguments that were set to default from the YAML file.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # define model
cfg.MODEL.WEIGHTS = "file"  # SET UP WEIGHTS HERE
cfg.MODEL.DEVICE = 'cpu'
# 5 classes (5 columns in this instance, but you may have more depending on what you are doing)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
# set the testing threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
predictor = DefaultPredictor(cfg)

# FUNCTIONS
# This function returns a list of vertical lines found within the image passed to the function.


# this function takes as parameter an image and default integers. It returns a list.
def get_vertical_lines(img, width=385, line_height=2000, circle=155):
    '''This function takes an image and default integers as parameters and outputs a list.'''
    ys = []
    keepers = []
    n = 0
    # convert between RGB/BGR and grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use an Adaptive Thresholding approach where the threshold value = Gaussian weighted sum of the neighborhood values - constant value.
    # In other words, it is a weighted sum of the blockSize^2 neighborhood of a point minus the constant.
    # in this example, we are setting the maximum threshold value as 255 with the block size of 155 (as set in the "circle" parameter) and the
    # constant is 2 (as specified in the last argument)
    edges = ~cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, circle, 2)
    # create a 3x3 matrix of ones.
    # An image kernel is a small matrix used to apply effects like the ones you might find in Photoshop or Gimp, such as blurring, sharpening, outlining or embossing. They're also used in machine learning for 'feature extraction', a technique for determining the most important portions of an image.
    kernel = np.ones((3, 3), np.uint8)
    # The basic idea of erosion is just like soil erosion only, it erodes away the boundaries of foreground object (Always try to keep foreground in white). It is normally performed on binary images. It needs two inputs, one is our original image, second one is called structuring element or kernel which decides the nature of operation. A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
    th2 = cv2.erode(edges, kernel, iterations=1)
    kernel = np.ones((1, 7), np.uint8)
    th3 = cv2.dilate(th2, kernel, iterations=1)
    lines = cv2.HoughLines(th3, 1, np.pi/180, line_height)
    for line in range(len(lines)):
        if lines[line][0][1] > -.1 and lines[line][0][1] < .1:
            keepers.append(lines[line])
            n += 1
    for line2 in range(n):
        for rho, theta in keepers[line2]:
            b = np.sin(theta)
            y0 = b*rho
            a = np.cos(theta)
            x0 = a*rho
            x1 = int(x0 + 30*(-b))
            y1 = int(y0 + 30*(a))
            x2 = int(x0 - 30*(-b))
            y2 = int(y0 - 30*(a))
            slope = (y2-y1) / (x2-x1)
            intercept = y1 - (slope * x1)
            side = slope * width + intercept
            ys.append(intercept)
            ys.append(side)
    return ys

# This function returns a list of horizontal lines found in the image passed into the function.


# this function takes as parameter and image and default integers. It returns a list.
def get_horizontal_lines(img, width=385, line_width=150, circle=155):
    ys = []
    keepers = []
    n = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts image to grayscale
    edges = ~cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, circle, 2)  # applies threshold on image
    kernel = np.ones((3, 3), np.uint8)
    th2 = cv2.erode(edges, kernel, iterations=1)
    kernel = np.ones((7, 1), np.uint8)
    th3 = cv2.dilate(th2, kernel, iterations=1)
    lines = cv2.HoughLines(th3, 1, np.pi/180, line_width)
    for line in range(len(lines)):
        if lines[line][0][1] > 1.45 and lines[line][0][1] < 1.7:
            keepers.append(lines[line])
            n += 1
    for line2 in range(n):
        for rho, theta in keepers[line2]:
            b = np.sin(theta)
            y0 = b*rho
            a = np.cos(theta)
            x0 = a*rho
            x1 = int(x0 + 30*(-b))
            y1 = int(y0 + 30*(a))
            x2 = int(x0 - 30*(-b))
            y2 = int(y0 - 30*(a))
            slope = (y2-y1) / (x2-x1)
            intercept = y1 - (slope * x1)
            side = slope * width + intercept
            ys.append(intercept)
            ys.append(side)
    return ys


def crop_bot(img, width=385, line_width_crop=300):
    temp = img[-50:, 0:width]
    try:
        ys = get_horizontal_lines(temp, line_width=line_width_crop)
        return img[:img.shape[0]-50+int(np.mean(ys)), 0:width]
    except:
        return img


def make_snippets(img, ys, rows=50, pixels_per_row=60, pixels_on_either_side=15, file_path="", column="lit", add_to_end=0):
    start = 0
    for y in range(rows):
        finish = start + pixels_per_row
        x_check = start - pixels_on_either_side
        x_check2 = start + pixels_on_either_side
        y_check = finish - pixels_on_either_side
        y_check2 = finish + pixels_on_either_side
        newlist = [x for x in ys if (x > x_check) & (x < x_check2)]
        newlist2 = [x for x in ys if (x > y_check) & (x < y_check2)]
        if len(newlist) != 0:
            start = round(statistics.median(newlist))
        if len(newlist2) != 0:
            finish = round(statistics.median(newlist2))
        if y == rows-1:
            snippet = img[start:]
        elif y != rows-1:
            snippet = img[start:finish]
        start = finish
        cv2.imwrite(file_path + "_" + column +
                    "_row_" + str(y+1) + ".jpg", snippet)


# CODE THAT DOES THE SEGMENTATION
bad = []
files = os.listdir()
# files = random.sample(os.listdir(), 4)
for d in files:
    if d[-4:] == ".jpg":
        try:
            out_dir = "/home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/segments/snippets/{}".format(
                numdir + "/" + album)
            im = cv2.imread(d)
            outputs = predictor(im)
            objects = outputs["instances"].pred_classes
            boxes = outputs["instances"].pred_boxes
            masks = outputs["instances"].pred_masks
            boxes_np = boxes.tensor.cpu().numpy()
            obj_np = objects.cpu().numpy()
            masks_np = masks.cpu().numpy()
            m = 0
            for box in range(len(boxes_np)):
                left = int(boxes_np[box][0])
                top = int(boxes_np[box][1])
                right = int(boxes_np[box][2])
                bottom = int(boxes_np[box][3])
                cropped_array = im[top:bottom, left:right]
                mask = masks_np[m][top:bottom, left:right]
                h, w = mask.shape
                tl = int(np.argwhere(mask[200] == True)[0])
                bl = int(np.argwhere(mask[h-200] == True)[0])
                white1 = np.zeros([h, w, 3], dtype=np.uint8)
                white1.fill(255)
                white2 = np.zeros([h, w, 3], dtype=np.uint8)
                white2.fill(255)
                change = (tl-bl)/h
                white3 = (cropped_array *
                          mask[..., None]) + (white1 * ~mask[..., None])
                for i in range(h):
                    start = int(tl - i*change)
                    if len(np.argwhere(mask[i] == True)) > 0:
                        last = int(np.argwhere(mask[i] == True)[-1])
                    elif len(np.argwhere(mask[i] == True)) == 0:
                        last = w-start
                    white2[i][0:last-start] = white3[i][start:last]
                if obj_np[m] == 0:
                    white3 = white2[:, 0:60]
                    outputs2 = predictor2(white3)
                    boxes2 = outputs2["instances"].pred_boxes
                    boxes_np2 = boxes2.tensor.cpu().numpy()
                    bottom2 = int(boxes_np2[0][3])
                    no_top = white3[bottom2:, :]
                    no_bot_or_top = crop_bot(
                        no_top, width=60, line_width_crop=45)
                    no_bot_or_top = cv2.resize(no_bot_or_top, (60, 3000))
                    ys = get_horizontal_lines(
                        no_bot_or_top, width=60, line_width=45)
                    make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,
                                  pixels_on_either_side=15, file_path=out_dir + "/" + d[:-4], column='lit1')
                elif obj_np[m] == 1:
                    white3 = white2[:, 0:60]
                    outputs2 = predictor2(white3)
                    boxes2 = outputs2["instances"].pred_boxes
                    boxes_np2 = boxes2.tensor.cpu().numpy()
                    bottom2 = int(boxes_np2[0][3])
                    no_top = white3[bottom2:, :]
                    no_bot_or_top = crop_bot(
                        no_top, width=60, line_width_crop=45)
                    no_bot_or_top = cv2.resize(no_bot_or_top, (60, 3000))
                    ys = get_horizontal_lines(
                        no_bot_or_top, width=60, line_width=45)
                    make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,
                                  pixels_on_either_side=15, file_path=out_dir + "/" + d[:-4], column='lit2')
                elif obj_np[m] == 2:
                    white3 = white2[:, 0:60]
                    outputs2 = predictor2(white3)
                    boxes2 = outputs2["instances"].pred_boxes
                    boxes_np2 = boxes2.tensor.cpu().numpy()
                    bottom2 = int(boxes_np2[0][3])
                    no_top = white3[bottom2:, :]
                    no_bot_or_top = crop_bot(
                        no_top, width=60, line_width_crop=45)
                    no_bot_or_top = cv2.resize(no_bot_or_top, (60, 3000))
                    ys = get_horizontal_lines(
                        no_bot_or_top, width=60, line_width=45)
                    make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,
                                  pixels_on_either_side=15, file_path=out_dir + "/" + d[:-4], column='lang1')
                elif obj_np[m] == 3:
                    white3 = white2[:, 0:350]
                    outputs2 = predictor2(white3)
                    boxes2 = outputs2["instances"].pred_boxes
                    boxes_np2 = boxes2.tensor.cpu().numpy()
                    bottom2 = int(boxes_np2[0][3])
                    no_top = white3[bottom2:, :]
                    no_bot_or_top = crop_bot(no_top, line_width_crop=265)
                    no_bot_or_top = cv2.resize(no_bot_or_top, (350, 3000))
                    ys = get_horizontal_lines(
                        no_bot_or_top, width=350, line_width=265)
                    make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,
                                  pixels_on_either_side=15, file_path=out_dir + "/" + d[:-4], column='lang2')
                elif obj_np[m] == 4:
                    white3 = white2[:, 0:225]
                    outputs2 = predictor2(white3)
                    boxes2 = outputs2["instances"].pred_boxes
                    boxes_np2 = boxes2.tensor.cpu().numpy()
                    bottom2 = int(boxes_np2[0][3])
                    no_top = white3[bottom2:, :]
                    no_bot_or_top = crop_bot(no_top, line_width_crop=300)
                    no_bot_or_top = cv2.resize(no_bot_or_top, (225, 3000))
                    ys = get_horizontal_lines(
                        no_bot_or_top, width=225, line_width=150)
                    make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,
                                  pixels_on_either_side=15, file_path=out_dir + "/" + d[:-4], column='rel')
                m += 1
        except:
            bad.append(d)
            traceback.print_exc()
            print("image failed: " + d)
            pass

print("Percent Error: " + str(len(bad)/len(files)))
print(bad)
with open(f'/home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/error_img/mexico_error_{numdir}.csv', 'a') as output:
 # /home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/error_img/mexico_error_62.csv
    # ../../../../error_img
    writer = csv.writer(output, delimiter=',')
    writer.writerow(bad)
