from flask import Flask, Response, url_for
from flask_cors import CORS
import cv2
from keras import models, layers
from collections import deque
import numpy as np


app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)
Q = deque(maxlen=32)
model = models.load_model('modelnew.h5')

layer = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')

def detect_fight(frame):
    writer = None
    (W, H) = (None, None)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame = frame.reshape(128, 128, 3) / 255

    # make predictions on the frame and then update the predictions
    # queue
    predictions = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(predictions)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = (predictions > 0.50)[0]
    label = i

    return "Violence: {}".format(label)


def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect fight in the frame
            fight_detected_text = detect_fight(frame)

            # Draw detection result on the frame

            cv2.putText(frame, fight_detected_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the output frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static')
def static_path():
    return url_for('static', filename='style.css')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
