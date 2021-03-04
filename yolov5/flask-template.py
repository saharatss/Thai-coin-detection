import flask
import cv2
import numpy as np


def main_process(image):
	return image



# region Flask
app = flask.Flask(__name__)
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
	def __del__(self):
		self.video.release()
	def get_frame(self):
		_,img = self.video.read()
		# img = np.zeros((10,10,3))
		# img	= cv2.resize(img,None,fx=0.5,fy=0.5)
		# img = cv2.putText(img, "img", (10,45), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,0,255), 2)
		img = main_process(img)
		ret, jpeg = cv2.imencode('.jpg', img)
		return jpeg.tobytes()
def gen(camera):
	while True:
		img = camera.get_frame()
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n')

@app.route('/')
def video_feed():
	return flask.Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(host='0.0.0.0',port='5000', debug=True)
# endregion
