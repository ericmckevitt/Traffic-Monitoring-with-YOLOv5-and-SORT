import cv2
import numpy as np
import torch 
import os 
from sort import *

# Function to perform Non-Maximum Suppression
def non_max_suppression(boxes, scores, overlapThresh):
    if len(boxes) == 0:
        return []

    # Convert bounding boxes to float numpy array
    boxes = np.array(boxes, dtype="float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = x1 + boxes[:,2]
    y2 = y1 + boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype("int")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.float()
model.eval()

savepath = os.path.join(os.getcwd(), 'data', 'video') 

USE_WEBCAM = False  # Set to False to use the MP4 file

# Conditional video source selection
if USE_WEBCAM:
    vid = cv2.VideoCapture(0)  # Use webcam
else:
    video_path = 'red_light_running_compilation.mp4'  # Path to your MP4 file
    vid = cv2.VideoCapture(video_path)  # Use MP4 file

mot_tracker = Sort() # create instance of the SORT tracker 

colours = [(255, 0, 0),   # Blue
           (0, 255, 0),   # Green
           (0, 0, 255),   # Red
           (255, 255, 0), # Cyan
           (255, 0, 255), # Magenta
           (0, 255, 255), # Yellow
           (255, 255, 255)] # White

colours = colours * (100 // len(colours) + 1)

car_colour = (255, 0, 0)   # Red for cars
human_colour = (0, 255, 0) # Green for humans

# Initialize a dictionary to map SORT IDs to class IDs
sort_id_to_class_id = {}

# Initialize a dictionary to store the positions of each tracked object
tracked_positions = {}

# Load the template image for the red traffic light
template = cv2.imread(os.path.join('img', 'red_light.png'), 0) 
w, h = template.shape[::-1]

while(True):
    ret, image_show = vid.read()

    # Preprocessing the frame
    # Gaussian Blurring
    blurred_frame = cv2.GaussianBlur(image_show, (5, 5), 0)

    # Histogram Equalization (in grayscale)
    gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)

    # Color Filtering (Optional)
    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    # Define range for red color and create mask
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
    # Range for upper range of red
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)
    # Generating the final mask
    red_mask = mask1 + mask2

    # Combining the mask with the equalized frame
    target_frame = cv2.bitwise_and(equalized_frame, equalized_frame, mask=red_mask)

    # Perform template matching
    res = cv2.matchTemplate(target_frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.08  # adjust this threshold as needed
    loc = np.where(res >= threshold)

    # Collect all detections with their match values
    detections = []
    match_values = []
    for pt in zip(*loc[::-1]):
        match_value = res[pt[1], pt[0]]
        detections.append((pt[0], pt[1], w, h))
        match_values.append(match_value)

    # Apply Non-Maximum Suppression
    boxes = non_max_suppression(detections, match_values, overlapThresh=0.3)

    # Draw bounding box for each detected red traffic light
    for (startX, startY, width, height) in boxes:
        cv2.rectangle(image_show, (startX, startY), (startX + width, startY + height), (0, 0, 255), 2)
        cv2.putText(image_show, "Red Traffic Light", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    preds = model(image_show)

    # Filter out detections to only include cars (class index 2) and humans (class index 0)
    filtered_detections = []
    for det in preds.pred[0]:
        if int(det[-1]) in [0, 2]:  # Check class ID
            bbox = det[:4].tolist()
            score = det[4].item()
            cls_id = int(det[5].item())
            filtered_detections.append([*bbox, score, cls_id])

    # Convert to numpy array for SORT
    if filtered_detections:
        filtered_detections_np = np.array(filtered_detections)
        track_bbs_ids = mot_tracker.update(filtered_detections_np[:,:4]) # Only pass bbox coordinates to SORT
        
        # Update SORT ID to class ID mapping
        for det, track in zip(filtered_detections, track_bbs_ids):
            sort_id_to_class_id[int(track[4])] = det[5]
    else:
        track_bbs_ids = np.empty((0, 5))

    print("Tracked objects:")
    for track in track_bbs_ids:
        print(track)

    for j in range(len(track_bbs_ids)):
        coords = track_bbs_ids[j]
        x1, y1, x2, y2, obj_id = map(int, coords[:5])

        # Calculate the centroid of the bounding box
        centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # Update the tracked positions
        if obj_id not in tracked_positions:
            tracked_positions[obj_id] = [centroid]
        else:
            tracked_positions[obj_id].append(centroid)

        # Draw the path of the tracked object
        for k in range(1, len(tracked_positions[obj_id])):
            if k == 1:
                continue
            cv2.line(image_show, tracked_positions[obj_id][k - 1], tracked_positions[obj_id][k], color, 2)

        # Retrieve class ID from the mapping
        cls_id = sort_id_to_class_id.get(obj_id, None)
        
        # Draw bounding box and label if class ID is known
        if cls_id is not None:
            if cls_id == 0:  # Human
                color = human_colour
                label = "Human"
            elif cls_id == 2:  # Car
                color = car_colour
                label = "Car"
            else:
                continue  # Skip if it's not a car or human

            cv2.rectangle(image_show, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_show, f"{label} ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the image
    cv2.imshow('Image', image_show)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
vid.release()
cv2.destroyAllWindows()
