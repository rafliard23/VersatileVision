import cv2
import numpy as np

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
		
	# initialize the list of picked indexes	
	pick = []
	
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
		
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

cap = cv2.VideoCapture('walk1.mp4')

## Create object detection from stable camera
# Create background substraction

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) 

while True:
    ret, frame = cap.read()

    ## Extract Region of Interest (minimalisir processing)
    # ambil RoI dengan "crop" gambar dari frame
    roi = frame

    # Add masking
    mask = object_detector.apply(roi)
    # Filter masking
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_TOZERO)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty array for detection

    detections = []

    for cnt in contours:
        # Calculate area and small elements
        area = cv2.contourArea(cnt)
        if area > 150:
            # Menampilkan kontur dari masking yang di dapat.
            #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 1)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

            detections.append([x, y, w, h])

            non_max_suppression_fast(detections, 0.2)

            # print for debug
            #print(detections)
    
    cv2.imshow("Frame", mask)
    # cv2.imshow("Masking", mask)

    key = cv2.waitKey(16)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 