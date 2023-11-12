import cv2
import numpy as np
import torch 
import os 
from sort import *

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

while(True):
    ret, image_show = vid.read()
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
