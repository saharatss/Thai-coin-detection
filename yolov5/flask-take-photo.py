import cv2
import numpy as np
import datetime, time
import random

import flask
app = flask.Flask(__name__)

file_dir = '/Users/saharatsaengsawang/Desktop/2020/yolo/coin_dataset/images/val/'

global_image = None

class VideoHttp(object):
	def get_frame(self):
		global global_image
		self.video = cv2.VideoCapture('http://172.20.10.1/live')
		_,img = self.video.read()
		global_image = img.copy()
		self.video.release()
		return img

def image_generator(video_feeder):
	while True:
		image = video_feeder.get_frame()
		_,jpeg = cv2.imencode('.jpg', image)
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def video_feed():
	return flask.Response(image_generator(VideoHttp()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take')
def take_picture():
	global global_image
	if global_image is None: return str(global_image)
	filename = str(random.randrange(100000, 999999))+'.jpeg'
	cv2.imwrite(file_dir+filename,global_image)
	return filename

if __name__ == '__main__':
	app.run(host='0.0.0.0',port='5000', debug=True)
# endregion