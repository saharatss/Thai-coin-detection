import cv2
import numpy as np
import datetime

my_confidence = 0.5
my_threshold  = 0.3

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "yolov3.weights"
configPath	= "yolov3.cfg"
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

def main_process(image):
	(H, W) = image.shape[:2]
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > my_confidence:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, my_confidence, my_threshold)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			image = cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, color, 2)

	return image



# region Flask
import flask
app = flask.Flask(__name__)
class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
	def __del__(self):
		self.video.release()
	def get_frame(self):
		fps_start = datetime.datetime.now()
		_,img = self.video.read()
		# img = np.zeros((10,10,3))
		# img	= cv2.resize(img,None,fx=0.5,fy=0.5)
		img = main_process(img)
		fps_end = datetime.datetime.now()
		fps_interval = 1/(fps_end-fps_start).total_seconds()
		img = cv2.putText(img, "fps "+str(round(fps_interval,2)), (10,45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2)
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
