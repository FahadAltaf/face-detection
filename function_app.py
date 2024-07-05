import azure.functions as func
import logging
import cv2
import numpy as np
import tempfile
import requests
import json

app = func.FunctionApp()

# Load the pre-trained DNN face detection model
model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
config_file = "models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

def detect_faces(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x2, y2) = box.astype("int")
            w, h = x2 - x, y2 - y
            faces.append({
                'index': int(i), 
                'x': int(x), 
                'y': int(y), 
                'width': int(w), 
                'height': int(h)
            })  # Ensure all values are standard Python ints
    return faces

@app.route(route="FaceDetectionFunction", auth_level=func.AuthLevel.FUNCTION)
def FaceDetectionFunction(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        image_url = req_body.get('image_url')
        
        if not image_url:
            return func.HttpResponse("No image URL provided", status_code=400)

        # Download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            return func.HttpResponse(f"Failed to download image from URL: {image_url}", status_code=400)

        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(response.content)
        temp_file.close()

        # Detect faces
        faces = detect_faces(temp_file.name)

        # Return the result
        result = {
            'filename': image_url,
            'faces': faces
        }
        return func.HttpResponse(json.dumps(result), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        return func.HttpResponse(f"Error processing the image: {e}", status_code=500)
