import requests
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import base64
import json

# # Define URL of Flask app endpoint
url = 'http://localhost:5000/predict'

# Load example image
image = Image.open("test_img.jpg")

# Open image file and send POST request with file parameter
with open("test_img.jpg", 'rb') as file:
    response = requests.post(url, files={'image': file})

# Print prediction
print(response.json())
# print(response.json()['prediction'])
# print(response.json()['bbox'])
# print(response.json()['landmarks'])

# Send image to Flask route
import json

url = 'http://localhost:5000/annotate_image'
landmarks = response.json()['landmarks']

payload = {"landmarks": json.dumps(landmarks)}
# print(landmarks)

response = requests.post(url, data=payload)

with open("test_img.jpg", 'rb') as file:
    response = requests.post(url, files={'image': file}, data=payload)
       
img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Define URL of Flask app endpoint
url = 'http://localhost:5000/predict_and_annotate'

# Open image file and send POST request with file parameter
with open("test_img.jpg", 'rb') as file:
    response = requests.post(url, files={'image': file})

# Get data
data = response.json()
print(data['prediction'])
# print(data['image'])
img_bytes = base64.b64decode(data['image'])
img_array = np.frombuffer(img_bytes, dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()