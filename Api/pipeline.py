import os 
import random
import time
from tqdm import tqdm

import tensorflow as tf
import cv2
from PIL import Image, ImageOps

import pandas as pd
import numpy as np
import mediapipe as mp

net = cv2.dnn.readNetFromDarknet('../Assets/yolov3.cfg', '../Assets/yolov3.weights')
classes = ['person']

def normalize_bbox(bbox, image_width, image_height):
    """
    Normalize the bounding box coordinates in YOLO style.

    Parameters:
        bbox (tuple): A tuple containing (x, y, w, h) of the bounding box coordinates.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        Tuple: A tuple containing normalized (x, y, w, h) coordinates of the bounding box.
    """
    x, y, w, h = bbox
    x_normalized = x / image_width
    y_normalized = y / image_height
    w_normalized = w / image_width
    h_normalized = h / image_height
    return x_normalized, y_normalized, w_normalized, h_normalized


def crop_image(image):
    """
    Detects people in images using YOLOv3 and crops the images to include only the people.
    
    Args:
        image (np.array): Numpy Array.
    """           
    try:
        frame = image
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)
        
    except Exception as e:
        print(f"Error in Image")
        print(e)
        return None, None, None
        
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and class_id < len(classes) and classes[class_id] == 'person':
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (center_x, center_y, width, height) = box.astype('int')
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                # increase box size by a scaling factor
                scaling_factor = 1.25  # change the scaling factor to 1.25
                w = int(width * scaling_factor)
                h = int(height * scaling_factor)
                # adjust box coordinates
                x = max(0, int(center_x - (w / 2)))
                y = max(0, int(center_y - (h / 2)))
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            cropped_frame = frame[y:y+h, x:x+w]
            normalized_bbox = normalize_bbox(box, frame.shape[1], frame.shape[0])
            return cropped_frame, normalized_bbox, box

# set up MediaPipe pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

features = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']


# function to extract pose landmarks from an image
def extract_pose_landmarks(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image)
        if results.pose_landmarks is not None:
            landmarks = [[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark]
            landmarks_dict = dict(zip(features, landmarks))
        else:
            landmarks = np.zeros(33*3)
            landmarks_dict = dict(zip(features, landmarks))
        return landmarks_dict

def pad_and_resize_image(image, desired_size):

    image = Image.fromarray(image)
    
    # Determine the desired size of the padded image
    padded_size = max(image.size)

    # Calculate the amount of padding needed
    delta_w = padded_size - image.size[0]
    delta_h = padded_size - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    # Add the padding to the image
    padded_image = ImageOps.expand(image, padding, fill=0)

    # Resize the padded image to the desired size
    resized_image = padded_image.resize((desired_size, desired_size))

    return np.array(resized_image)

model = tf.keras.models.load_model('../models/vision_model.h5')


def pipeline(image):
    image = image.copy()
    cropped_img, normalized_bbox, box = crop_image(image)

    if cropped_img is None:
        print("No person detected in image")
        return None, None, None

    landmarks_dict = extract_pose_landmarks(image)

    padded_img = pad_and_resize_image(cropped_img, 224)
    
    pred = model.predict(padded_img.reshape(1, 224, 224, 3), verbose=0)
    pred = np.argmax(pred, axis=1)

    poses = ['Bhujangasana', 'Padmasana', 'Shavasana', 'Tadasana', 'Trikonasana', 'Vrikshasana']
    # return poses[pred[0]], normalized_bbox, landmarks_dict
    return poses[pred[0]], normalized_bbox, landmarks_dict

def benchmark():
    # Load the image once
    img_files = os.listdir("D:/Hashane/MSC/Module/research 2022/work/code/New folder/1/Client-Yoga-Pose-SGCNN/Dataset/Extracted_Images/Abhay_Bhujangasana/")
    # Run the code 50 times and time it
    total_time = 0
    num_iterations = 50

    for i in range(num_iterations):
        start_time = time.time()
        
        # Load the image
        img = cv2.imread("D:/Hashane/MSC/Module/research 2022/work/code/New folder/1/Client-Yoga-Pose-SGCNN/Dataset/Extracted_Images/Abhay_Bhujangasana/" + img_files[i])

        # Run the code
        try:
            result = pipeline(img)
        except Exception as e:
            # print(e)
            continue
        # print(result[1], flush=True)
        
        end_time = time.time()
        
        iteration_time = end_time - start_time
        total_time += iteration_time
        
        print(f"Iteration {i+1}: {iteration_time:.4f} seconds")

    # Calculate the average time
    average_time = total_time / num_iterations
    print(f"Average time: {average_time:.4f} seconds")

if __name__ == "__main__":
    
    img = cv2.imread("D:/Hashane/MSC/Module/research 2022/work/code/New folder/1/Client-Yoga-Pose-SGCNN/Dataset/Extracted_Images/Abhay_Bhujangasana/Abhay_Bhujangasana_image_5.1s.jpg")
    
    # print(pipeline(img))
    benchmark()

    # cropped_img, box = crop_image(img)

    # if cropped_img is None:
    #     print("No person detected in image")
    #     return None

    # padded_img = pad_and_resize_image(cropped_img, 224)
    
    # pred = model.predict(padded_img.reshape(1, 224, 224, 3), verbose=0)
    # pred = np.argmax(pred, axis=1)

    # poses = ['Bhujangasana', 'Padmasana', 'Shavasana', 'Tadasana', 'Trikonasana', 'Vrikshasana']
    # print(pred[0])
    # print(poses[pred[0]])

    # return poses[pred[0]]