#for video
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
#for text to sppech
from audioplayer import AudioPlayer
from gtts import gTTS

print("***********************LOADING Model**********************************")
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
#save all the names in file o the list classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the detectionput layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("***********************Model LOADED*************************************")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
# loop over the frames from the video stream
oldLabel=''
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	height, width, channels = frame.shape
	frame = imutils.resize(frame, width=400)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward(output_layers)
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for detection in detections:
		for detection_info in detection:
			scores = detection_info[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.7:
				# Object detected
				center_x = int(detection_info[0] * width)
				center_y = int(detection_info[1] * height)
				w = int(detection_info[2] * width)
				h = int(detection_info[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	#We use NMS function in opencv to perform Non-maximum Suppression
	#we give it score threshold and nms threshold as arguments.
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[class_ids[i]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			#a =  math.floor(w / 2)
			#cv2.line(frame, (x+a, y), (x+a ,y + h),color,1)
			cv2.putText(frame, label, (x, y-5),cv2.FONT_HERSHEY_SIMPLEX, 1/2, color, 2)
			#code for playing audio
			if  oldLabel != label :
				objectName = "{} is near to you".format(label)
				if x < 100 :
					objName = "{} is on your right".format(label)
				elif x > 300 :
					objName = "{} is on your left".format(label)
				else :
					objName = "{} is in front of you".format(label)
				print("{}".format(objName))    
				tts = gTTS(objName)
				tts.save('1.wav')
				soundfile='1.wav'
				AudioPlayer("1.wav").play(block=True)
				oldLabel = label
	cv2.imshow("Image",frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
    # stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
