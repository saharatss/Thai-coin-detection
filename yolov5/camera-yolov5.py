import cv2
import numpy as np
import datetime

import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
		plot_one_box, strip_optimizer, set_logging, increment_dir
from utils.torch_utils import select_device, load_classifier, time_synchronized

imgsz = 640
weights = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']

my_confidence 		= 0.80 # 0.25
my_threshold  		= 0.45 # 0.45
my_filterclasses 	= None
# my_weight					= './weights/yolov5s.pt'
my_weight					= './weights/coin_v1-7_last.pt'

set_logging()
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA
print('>> device',device.type)

# Load model
model = attempt_load(my_weight, map_location=device)	# load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())		# check img_size
if half:
	model.half()  # to FP16

# Second-stage classifier
classify = False
if classify:
	modelc = load_classifier(name='resnet101', n=2)  # initialize
	modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
	modelc.to(device).eval()

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

def main_process(input_img):
	img0 = input_img.copy()

	img = letterbox(img0, new_shape=imgsz)[0]
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)

	img = torch.from_numpy(img).to(device)
	img = img.half() if half else img.float()
	img /= 255.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	t1 = time_synchronized()
	pred = model(img, augment=True)[0]
	pred = non_max_suppression(pred, my_confidence, my_threshold, classes=my_filterclasses, agnostic=None)
	t2 = time_synchronized()

	total = 0
	for i, det in enumerate(pred):
		gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
		if det is not None and len(det):
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
			for *xyxy, conf, cls in reversed(det):
				xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
				label = '%sbaht (%.0f%%)' % (names[int(cls)], conf*100)
				total += int(names[int(cls)])
				plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
	print('Done. (%.3fs)' % (t2 - t1))
	
	return img0


cap = cv2.VideoCapture(0)

if __name__ == '__main__':
	while True:
		_,img = cap.read()
		img = main_process(img).copy()
		cv2.imshow('image',img)
		cv2.waitKey(1)

# endregion
