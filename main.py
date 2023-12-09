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
DEBUG = False  # Set to True to print debug messages

# Conditional video source selection
if USE_WEBCAM:
    vid = cv2.VideoCapture(0)  # Use webcam
else:
    video_path = 'videos/red_light_running_compilation.mp4'  # Path to your MP4 file
    vid = cv2.VideoCapture(video_path)  # Use MP4 file

mot_tracker = Sort() # create instance of the SORT tracker 

# Load the red light template
red_light_template = cv2.imread(os.path.join('img', 'red_light4.png'), 0) 
template_h, template_w = red_light_template.shape[:2]

def detect_red_lights(frame, template):
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the template to grayscale only if it has more than 1 channel
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template  # Template is already in grayscale

    # Perform template matching
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.37
    loc = np.where(result >= threshold)
    is_red_light_detected = np.any(result >= threshold)
    return loc, is_red_light_detected


colours = [(255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 255, 0), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 255), # Yellow
            (255, 255, 255)] # White

colours = colours * (100 // len(colours) + 1)

car_colour = (255, 0, 0)   # Blue for cars

# Initialize a dictionary to map SORT IDs to class IDs
sort_id_to_class_id = {}

# Initialize a dictionary to store the positions of each tracked object
tracked_positions = {}

def mouse_click(event, x, y, flags, param):
    global line_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(line_points) < 2:
            line_points.append((x, y))
            print("Point added:", (x, y))

def compute_line_equation(p1, p2):
    # Compute coefficients A, B, and C for the line equation Ax + By + C = 0
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = p2[0] * p1[1] - p1[0] * p2[1]
    return A, B, C

def check_side_of_line(point, line_coeff):
    # Check which side of the line the point is on
    # Returns positive if on one side, negative if on the other, and 0 if on the line
    A, B, C = line_coeff
    x, y = point
    return A*x + B*y + C

# Preliminary loop to select the line
line_points = []
while len(line_points) < 2:
    ret, image_show = vid.read()
    if not ret:
        break

    # Show the first point if it's selected
    if len(line_points) == 1:
        cv2.circle(image_show, line_points[0], 5, (0, 255, 0), -1)

    cv2.imshow('Image', image_show)
    cv2.setMouseCallback('Image', mouse_click)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Compute the line equation now that we have both points
line_coeff = compute_line_equation(line_points[0], line_points[1])

while(True):
    ret, image_show = vid.read()

    # Detect red lights in the current frame
    red_light_locations, is_red_light_detected = detect_red_lights(image_show, red_light_template)

    if is_red_light_detected:
        cv2.putText(image_show, "Red light", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # Draw the line that car has to cross to register red light violation
    if len(line_points) == 2:
        cv2.line(image_show, line_points[0], line_points[1], (0, 255, 0), 2)

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

    if DEBUG: 
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
            if cls_id == 2:  # Car
                color = car_colour
                label = "Car"
            else:
                continue  # Skip if it's not a car

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
