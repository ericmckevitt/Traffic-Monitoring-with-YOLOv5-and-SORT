import cv2
import numpy as np
import torch 
import os 
from sort import *

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.float()
model.eval()

savepath = os.path.join(os.getcwd(), 'data', 'video') 
vid = cv2.VideoCapture(0)

mot_tracker = Sort() # create instance of the SORT tracker 

colours = [(255, 0, 0),   # Blue
           (0, 255, 0),   # Green
           (0, 0, 255),   # Red
           (255, 255, 0), # Cyan
           (255, 0, 255), # Magenta
           (0, 255, 255), # Yellow
           (255, 255, 255)] # White

colours = colours * (100 // len(colours) + 1)

while(True):

    ret, image_show = vid.read()
    preds = model(image_show)
    detections = preds.pred[0].numpy()
    track_bbs_ids = mot_tracker.update(detections)

    for j in range(len(track_bbs_ids.tolist())):

        coords = list(track_bbs_ids)[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = f"ID: {name_idx}"
        color = colours[name_idx]
        cv2.rectangle(image_show, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_show, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('Image', image_show)

    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit
        break

vid.release()
cv2.destroyAllWindows()