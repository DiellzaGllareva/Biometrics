import numpy as np
import cv2
import os

# load face detector model
print("[INFO] loading face detector...")
protoPath = os.path.join("face_detector", "deploy.prototxt")
modelPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#load video stream and point to the location of the video
vs = cv2.VideoCapture("videos\\fake.mp4")
read = 0
saved = 0

# loop over frames from the video file stream
while True:
	# grab the frame from the file
	(grabbed, frame) = vs.read()

	if not grabbed:
		break

	# increment the total number of frames read thus far
	read += 1

	# check to see if we should process this frame
	if read % 16 != 0:
		continue

	# grab the frame dimensions and construct a blob from the frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and predict
	net.setInput(blob)
	detections = net.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]

			# save image with label
			p = os.path.sep.join(["videos",
				"{}.png".format(saved)])
			cv2.imwrite(p, face)
			saved += 1
			print("[INFO] saved {} to disk".format(p))

vs.release()
cv2.destroyAllWindows()