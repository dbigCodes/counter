import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import numpy as np
import logging


logging.basicConfig(filename='logs.log',level=logging.INFO)
def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
	args = vars(ap.parse_args())

	conf = json.load(open(args["conf"]))
	video_capture = cv2.VideoCapture(0)
	people_entered = 0
	lastTime = datetime.datetime.now()

	# allow the camera to warmup, then initialize the average frame
	print "[INFO] warming up..."
	time.sleep(conf["camera_warmup_time"])
	buff = 0
	avg = None
	# capture frames from the camera
	while True:
		ret, frame = video_capture.read()
		timestamp = datetime.datetime.now()
		buff += 1
		if buff > 1: #para lang ni xia na dli niya makuha ang first na frame
			frame = imutils.resize(frame, width=500)
			frame_height = np.size(frame, 0) 
			frame_width = np.size(frame,1)
			counting_line = frame_height / 2 
			cv2.line(frame,(100,counting_line),(400,counting_line),(200,200,0),2) #drawing og counting line
			gray=convert_and_blur(frame) #fucntion call

			#initialize avg
			if avg is None:
				print "[INFO] starting background model..."
				avg = gray.copy().astype("float")
				continue

			thresh = accumulate_and_thresholding(gray, avg, conf)
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
	
			# loop over the contours
			for c in cnts:
				# pag gamay ang area kay e ignore lang
				if cv2.contourArea(c) < conf["min_area"]:
					continue

				
				(x, y, w, h) = cv2.boundingRect(c)
			
				if w > 25 and h > 25:
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
					x1 = w/2   #width of the bounding box   
					y1 = h/2   #width of the bounding box
					cx = x+x1
					cy = y+y1
					#getting the center of mass
					centroid = (cx,cy)
					#circle ni xia sa sulod sa rectangle
					cv2.circle(frame,(centroid),1,(0,0,255),1)
				
					if ((cx > 100) and (cx < 400)):
						if ((cy < counting_line+20) and (cy > counting_line-20)):
							if (timestamp - lastTime).seconds >= conf["min_seconds"]:
								#counting line becomes color red
								cv2.line(frame,(100,counting_line),(400,counting_line),(0,0,255),2)
								people_entered += 1
							if people_entered >= conf["min_people_entered"]:
								lastTime = timestamp

			cv2.putText(frame, "People Entered: {}".format(people_entered), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 121), 2)
			cv2.imshow("frame",frame)
			cv2.imshow("threshold",thresh)
	
		key = cv2.waitKey(30) & 0xff
		if key == 27:
			logging.info('The total of people entered is ' + `people_entered` + ' on ' + datetime.datetime.now().strftime("%A %d %B %Y"))
			break
	return 0

def convert_and_blur(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0) 
	return gray
def accumulate_and_thresholding(gray, avg, conf):
	kernel = np.ones((5,5),np.uint8)
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
	cv2.imshow('Frame Difference', frameDelta)
	thresh = cv2.threshold(frameDelta, conf["threshold"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	thresh = cv2.dilate(thresh, None, iterations=5)
	return thresh


if __name__ == "__main__":
	main()