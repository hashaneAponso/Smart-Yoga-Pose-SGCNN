from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from pipeline import pipeline
from draw_utils import draw_bbox_landmarks
import json
import traceback

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET'])
def test():
    return "App running!"


@app.route('/predict', methods=['POST'])
def predict():
    # Get image file from POST request
    image_file = request.files['image']

    # Load image file with PIL
    image = Image.open(image_file)

    # Convert PIL image to NumPy array
    image_array = np.asarray(image)
    # imgdata = request.args.get('image')

    # decoded = base64.b64decode(imgdata)
    # image_array = np.array(Image.open(BytesIO(decoded)))


    # Call pipeline function to get prediction
    try:
        result = pipeline(image_array)
        print(result)
        prediction, normalized_bbox, landmarks_dict = result
    except Exception as e:
        traceback.print_exc()
        return jsonify({"prediction": "", "bbox": "", "landmarks": ""})

    # Return prediction as JSON response
    return jsonify({"prediction": prediction, "bbox": normalized_bbox, "landmarks": landmarks_dict})

@app.route('/annotate_image', methods=['POST'])
def annotate_image():
    # Get image file from POST request
    image_file = request.files['image']
    landmarks = json.loads(request.form.get('landmarks'))
    # print("landmarks: ", landmarks, flush=True)

    # Load image file with PIL
    image = Image.open(image_file)

    # Convert PIL image to NumPy array
    image_array = np.asarray(image)

    if (landmarks == None) or (landmarks == ""):
        img = Image.fromarray(image)
        # save PIL Image to disk
        img.save('annotated_image.png')
        return send_file("annotated_image.png", mimetype='image/png', as_attachment=True)

    # print("image: ", image_array, flush=True)

    # Annotate Image
    try:
        annotated_image = draw_bbox_landmarks(image_array, bbox=[], landmarks=landmarks)
    except Exception as e:
        traceback.print_exc()

    img = Image.fromarray(annotated_image)

    # save PIL Image to disk
    img.save('annotated_image.png')

    # Send Image
    return send_file("annotated_image.png", mimetype='image/png', as_attachment=True)


@app.route('/predict_and_annotate', methods=['POST'])
def predict_and_annotate():
    # Get image file from POST request
    image_file = request.files['image']

    # Load image file with PIL
    image = Image.open(image_file)

    # Convert PIL image to NumPy array
    image_array = np.asarray(image)
    # imgdata = request.args.get('image')

    # decoded = base64.b64decode(imgdata)
    # image_array = np.array(Image.open(BytesIO(decoded)))


    # Call pipeline function to get prediction
    try:
        prediction, normalized_bbox, landmarks_dict = pipeline(image_array)
    except Exception as e:
        img = Image.fromarray(image_array)
        img_bytes = BytesIO()
        # save PIL Image to BytesIO
        img.save(img_bytes, format='PNG')
        return jsonify({"prediction": '', 'image': base64.b64encode(img_bytes.getvalue()).decode()})
        traceback.print_exc()

    # Annotate Image
    try:
        annotated_image = draw_bbox_landmarks(image_array, bbox=[], landmarks=landmarks_dict)
    except Exception as e:
        return jsonify({"prediction": '', 'image': base64.b64encode(image_file.read()).decode()})
        print(e, flush=True)

    img = Image.fromarray(annotated_image)
    img_bytes = BytesIO()
    # save PIL Image to BytesIO
    img.save(img_bytes, format='PNG')
    # img.save('annotated_image.png')

    # Send Image and Prediction
    return jsonify({"prediction": prediction, 'image': base64.b64encode(img_bytes.getvalue()).decode()})

if __name__ == '__main__':
    app.run()
