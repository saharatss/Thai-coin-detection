import cv2
import numpy as np
import datetime, time

import os
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer, set_logging, increment_dir
from utils.torch_utils import select_device, load_classifier, time_synchronized

imgsz = 640
weights = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']

my_confidence 		= 0.80 # 0.25
my_threshold  		= 0.45 # 0.45
my_filterclasses 	= None
# my_weight					= './weights/yolov5s.pt'
my_weight					= './weights/coin_v1-9_last.pt'


set_logging()
device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA
print('>> device',device.type)

# Load model
model = attempt_load(my_weight, map_location=device)	# load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())		# check img_size
if half:
	model.half()  # to FP16

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
colors = [
	(232, 182, 0), # 5Baht
	(0, 204, 255),	# 1Baht
	(69, 77, 246),	# 10Baht
	(51, 136, 222),	# 2Baht
	(222, 51, 188),	# .50Baht
	]

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
	class_count = [0 for _ in range(len(names))]
	for i, det in enumerate(pred):
		gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
		if det is not None and len(det):
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
			for *xyxy, conf, cls in reversed(det):
				class_count[int(cls)] += 1
				xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
				label = '%sbaht (%.1f%%)' % (names[int(cls)], conf*100)
				total += int(names[int(cls)])
				plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
				print(label)
	print('Done. (%.3fs)' % (t2 - t1))
	
	# cv2.rectangle(img0,(0,10),(250,90),(0,0,0),-1)
	img0 = cv2.putText(img0, "10Baht "+str(class_count[2])+" coin", (10,45+25*1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " 5Baht "+str(class_count[0])+" coin", (10,45+25*2), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " 2Baht "+str(class_count[3])+" coin", (10,45+25*3), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " 1Baht "+str(class_count[1])+" coin", (10,45+25*4), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200,200,0), 2)
	img0 = cv2.putText(img0, " Total "+str(total)+" Baht", 					(10,45+25*5), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,0), 2)
	
	return img0

cap = cv2.VideoCapture(0)
if __name__ == '__main__':
	fps_start = datetime.datetime.now()
	while(True):
		fps_start = datetime.datetime.now()
		_,img = cap.read()
		img = main_process(img).copy()
		fps_end = datetime.datetime.now()
		fps_interval = 1/(fps_end-fps_start).total_seconds()
		img = cv2.putText(img, "fps "+str(round(fps_interval,2)), (10,45+35*0), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
		cv2.imshow('Image',img)
		cv2.waitKey(1)

# region Flask
import flask
app = flask.Flask(__name__)

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		# self.video = cv2.VideoCapture('http://172.16.5.243')
	def __del__(self):
		self.video.release()
	def get_frame(self):
		fps_start = datetime.datetime.now()
		# img = cv2.imread('/Users/saharatsaengsawang/Desktop/2020/yolo/coin_dataset/images/set1_train/41.jpg')
		_,img = self.video.read()
		# img = np.zeros((10,10,3))
		# img	= cv2.resize(img,None,fx=0.5,fy=0.5)
		img = main_process(img).copy()
		fps_end = datetime.datetime.now()
		fps_interval = 1/(fps_end-fps_start).total_seconds()
		img = cv2.putText(img, "fps "+str(round(fps_interval,2)), (10,45+35*0), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
		return img

class VideoHttp(object):
	def get_frame(self):
		fps_start = datetime.datetime.now()
		self.video = cv2.VideoCapture('http://172.20.10.1/live')
		# self.video = cv2.VideoCapture('http://192.168.1.101/live')
		_,img = self.video.read()
		self.video.release()
		img = main_process(img).copy()
		fps_end = datetime.datetime.now()
		fps_interval = 1/(fps_end-fps_start).total_seconds()
		# img = cv2.putText(img, "fps  "+str(round(fps_interval,2)), (10,45+25*0), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,255), 2)
		return img

def image_generator(video_feeder):
	while True:
		image = video_feeder.get_frame()
		_,jpeg = cv2.imencode('.jpg', image)
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def video_feed():
	# return flask.Response(image_generator(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')
	return flask.Response(image_generator(VideoHttp()),mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
# 	app.run(host='0.0.0.0',port='5000', debug=True)
# endregion
